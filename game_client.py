"""
Геймтон DatsJingleBang - Клиент для игры
Стратегия: максимизация очков через эффективное уничтожение препятствий
"""

import requests
import time
import math
import random
import heapq
from typing import List, Dict, Tuple, Optional, Set

# Импорты моделей и модулей
from models import Position, Bomber, Bomb, Enemy, Mob, BoosterType
from pathfinding import PathFinder
from boosters import BoosterManager
from coordination import BomberCoordinator
from combat import CombatManager
from strategy import GameStrategy

# Продвинутая логика с симуляцией цепных реакций
from advanced_logic import MapPredictor, TargetSelector, SafePathfinder


class GameClient:
    """Клиент для игры"""
    
    def __init__(self, api_key: str, base_url: str = "https://games.datsteam.dev", use_local_api: bool = False):
        print(f"[CLIENT] Initializing GameClient with base_url={base_url}, use_local_api={use_local_api}")
        self.api_key = api_key
        self.base_url = base_url
        self.use_local_api = use_local_api
        
        if use_local_api:
            # Используем локальный веб-сервер
            from game_api import GameAPI
            self.api = GameAPI(api_key, base_url)
            self.base_url = "http://localhost:5000"
        else:
            # Прямое подключение к API
            self.session = requests.Session()
            self.session.headers.update({"X-Auth-Token": api_key})
            self.api = None
        
        # Состояние игры
        self.map_size: Optional[Tuple[int, int]] = None
        self.walls: Set[Position] = set()
        self.obstacles: Set[Position] = set()
        self.bombers: List[Bomber] = []
        self.bombs: List[Bomb] = []
        self.enemies: List[Enemy] = []
        self.mobs: List[Mob] = []
        
        # Статистика улучшений (будут управляться через BoosterManager)
        self.bomb_range = 1  # текущий радиус бомбы
        self.bomb_delay = 8000  # время до взрыва в мс
        
        # Rate limiting
        self.last_request_time = 0
        self.request_times = []
        self.backoff_until = 0  # Общий backoff после ошибки 429
        
        # Статистика для отслеживания изменений
        self.last_enemy_count = 0
        self.last_mob_count = 0
        self.last_obstacle_count = 0
        self.total_kills = 0
        self.total_obstacles_destroyed = 0
        
        # Продвинутая логика (инициализируется после первого update_state)
        self.map_predictor: Optional[MapPredictor] = None
        self.target_selector: Optional[TargetSelector] = None
        self.safe_pathfinder: Optional[SafePathfinder] = None
        self.use_advanced_logic = True  # Флаг для включения/выключения продвинутой логики
        
        # Инициализация модулей (будут созданы после первого update_state)
        self.path_finder: Optional[PathFinder] = None
        self.booster_manager: Optional[BoosterManager] = None
        self.coordinator: Optional[BomberCoordinator] = None
        self.combat_manager: Optional[CombatManager] = None
        self.strategy: Optional[GameStrategy] = None
    
    def _rate_limit(self):
        """Ограничение скорости запросов (1 запрос в 1.5 секунды для избежания 429)"""
        now = time.time()
        
        # Проверяем общий backoff
        if now < self.backoff_until:
            sleep_time = self.backoff_until - now
            print(f"[RATE_LIMIT] In backoff, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            now = time.time()
        
        # Ограничиваем частоту запросов: минимум 1.5 секунды между запросами
        self.request_times = [t for t in self.request_times if now - t < 2.0]
        if len(self.request_times) >= 1:
            sleep_time = 1.5 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
        
        self.request_times.append(now)
    
    def get_arena(self, retry_count: int = 3) -> Dict:
        """Получить состояние арены"""
        if self.use_local_api and self.api:
            return self.api.get_arena(use_cache=True)

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                response = self.session.get(f"{self.base_url}/api/arena")
                if response.status_code == 429:
                    # Устанавливаем общий backoff на 5 секунд
                    self.backoff_until = time.time() + 5.0
                    sleep_time = 5.0
                    print(f"[RATE_LIMIT] Got 429 in get_arena, setting backoff for {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                response.raise_for_status()
                # Сбрасываем backoff при успехе
                self.backoff_until = 0
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(2.0)
    
    def move_bombers(self, commands: List[Dict], retry_count: int = 2) -> Dict:
        """Отправить команды движения"""
        if not commands: return {}
        
        if self.use_local_api and self.api:
            result = self.api.move_bombers(commands)
            # Обрабатываем ошибки размещения бомб
            if result.get("errors"):
                self._handle_bomb_placement_errors(result["errors"], commands)
            return result

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                payload = {"bombers": commands}
                response = self.session.post(f"{self.base_url}/api/move", json=payload)
                if response.status_code == 429:
                    # Устанавливаем общий backoff на 5 секунд
                    self.backoff_until = time.time() + 5.0
                    sleep_time = 5.0
                    print(f"[RATE_LIMIT] Got 429 in move_bombers, setting backoff for {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                response.raise_for_status()
                result = response.json()
                # Сбрасываем backoff при успехе
                self.backoff_until = 0
                # Обрабатываем ошибки размещения бомб
                if result.get("errors"):
                    self._handle_bomb_placement_errors(result["errors"], commands)
                return result
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(2.0)
    
    def _validate_path(self, path: List[List[int]], bomber_pos: Position) -> List[List[int]]:
        """Валидировать путь: каждый шаг должен быть соседней клеткой
        
        Args:
            path: Путь в формате [[x, y], ...]
            bomber_pos: Текущая позиция юнита
            
        Returns:
            Валидный путь или пустой список если путь невалиден
        """
        if not path:
            return []
        
        # Проверяем, что путь начинается с текущей позиции или соседней клетки
        first_step = path[0]
        if len(first_step) != 2:
            return []
        
        first_pos = Position(first_step[0], first_step[1])
        
        # Первый шаг должен быть текущей позицией или соседней клеткой
        if first_pos != bomber_pos:
            dist = abs(first_pos.x - bomber_pos.x) + abs(first_pos.y - bomber_pos.y)
            if dist > 1:
                print(f"[PATH VALIDATION] Invalid first step: from {bomber_pos} to {first_pos} (distance={dist})")
                return []
        
        valid_path = [first_step]
        prev_pos = first_pos
        
        # Проверяем каждый последующий шаг
        for step in path[1:]:
            if len(step) != 2:
                print(f"[PATH VALIDATION] Invalid step format: {step}")
                break
            
            current_pos = Position(step[0], step[1])
            
            # Каждый шаг должен быть соседней клеткой (только по одной оси, не по диагонали!)
            dx = abs(current_pos.x - prev_pos.x)
            dy = abs(current_pos.y - prev_pos.y)
            
            # Проверяем, что шаг только по одной оси (не диагональный)
            if (dx > 1 or dy > 1) or (dx == 1 and dy == 1) or (dx > 0 and dy > 0):
                print(f"[PATH VALIDATION] Invalid step: from {prev_pos} to {current_pos} (dx={dx}, dy={dy}) - diagonal or too far")
                break
            
            # Проверяем, что позиция валидна
            if not self.is_valid_position(current_pos):
                print(f"[PATH VALIDATION] Invalid position in path: {current_pos}")
                break
            
            valid_path.append(step)
            prev_pos = current_pos
        
        return valid_path
    
    def _handle_bomb_placement_errors(self, errors: List[str], commands: List[Dict]):
        """Обработать ошибки размещения бомб и адаптироваться"""
        for error in errors:
            if "cannot place bomb" in error.lower():
                # Извлекаем ID бомбера и позицию из ошибки
                import re
                match = re.search(r'bomberman ([a-f0-9-]+) cannot place bomb at \[(\d+) (\d+)\]', error)
                if match:
                    bomber_id = match.group(1)
                    bomb_x, bomb_y = int(match.group(2)), int(match.group(3))
                    print(f"[BOMB ERROR] {bomber_id} cannot place bomb at ({bomb_x}, {bomb_y}) - removing from commands")
                    
                    # Удаляем бомбу из команд для этого бомбера
                    for cmd in commands:
                        if cmd.get("id") == bomber_id and cmd.get("bombs"):
                            # Удаляем бомбу, которая не может быть поставлена
                            cmd["bombs"] = [b for b in cmd["bombs"] if b != [bomb_x, bomb_y]]
                            if not cmd["bombs"]:
                                cmd.pop("bombs", None)
                            print(f"[BOMB ERROR] Removed invalid bomb placement for {bomber_id}")
            
            elif "invalid move" in error.lower():
                # Обрабатываем ошибки недопустимых ходов
                import re
                match = re.search(r'bomberman ([a-f0-9-]+) invalid move from \[(\d+) (\d+)\] to \[(\d+) (\d+)\]', error)
                if match:
                    bomber_id = match.group(1)
                    from_x, from_y = int(match.group(2)), int(match.group(3))
                    to_x, to_y = int(match.group(4)), int(match.group(5))
                    print(f"[MOVE ERROR] {bomber_id} invalid move from ({from_x}, {from_y}) to ({to_x}, {to_y})")
                    
                    # Удаляем невалидный путь из команд
                    for cmd in commands:
                        if cmd.get("id") == bomber_id and cmd.get("path"):
                            # Очищаем путь или оставляем только валидные шаги
                            cmd["path"] = []
                            print(f"[MOVE ERROR] Removed invalid path for {bomber_id}")
    
    def get_boosters(self, retry_count: int = 2) -> Optional[Dict]:
        """Получить доступные улучшения"""
        if self.use_local_api and self.api:
            try:
                return self.api.get_boosters()
            except Exception:
                return None

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                response = self.session.get(f"{self.base_url}/api/booster")
                if response.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if response.status_code == 400:
                    # Bad Request - возможно, игра еще не началась или нет доступа
                    return None
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: return None
                time.sleep(0.5)
        return None

    def is_bomber_enclosed(self, bomber: Bomber) -> bool:
        """Определить, окружён ли юнит препятствиями/стенами и не может выйти"""
        blocked = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = Position(bomber.pos.x + dx, bomber.pos.y + dy)
            if not self.is_valid_position(neighbor):
                blocked += 1
            elif neighbor in self.obstacles and (self.booster_manager.acrobatics_level if self.booster_manager else 0) < 2:
                blocked += 1
        return blocked >= 3
    
    def are_all_bombers_enclosed(self) -> bool:
        """Проверить, заперты ли все бомберы (нет выхода в радиусе или все окружены препятствиями)"""
        if self.coordinator:
            return self.coordinator.are_all_bombers_enclosed(
                self.bombers, self.obstacles, self.bomb_range, self.check_exit_within_radius
            )
        # Fallback
        alive_bombers = [b for b in self.bombers if b.alive]
        if not alive_bombers:
            return False
        first_bomber = alive_bombers[0]
        has_exit = self.check_exit_within_radius(first_bomber.pos, radius=5)
        if not has_exit and len(self.obstacles) > 0:
            for obs in self.obstacles:
                dist = first_bomber.pos.manhattan_distance(obs)
                if dist <= self.bomb_range + 3:
                    return True
        return not has_exit
    
    def is_space_truly_enclosed(self, bombers: List[Bomber]) -> bool:
        """Проверить, что пространство действительно замкнуто - нет пути к выходу"""
        if not bombers:
            return False
        
        alive_bombers = [b for b in bombers if b.alive]
        if not alive_bombers:
            return False
        
        # Проверяем, есть ли выход для хотя бы одного бомбера
        # Используем более быструю проверку: проверяем только позиции бомберов и их окрестности
        for bomber in alive_bombers:
            # Проверяем, есть ли выход в радиусе от позиции бомбера
            if self.check_exit_within_radius(bomber.pos, radius=10):
                return False  # Нашли выход - пространство не замкнуто
            
            # Также проверяем, можем ли мы дойти до свободного пространства
            # Проверяем несколько направлений
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                check_pos = Position(bomber.pos.x + dx * 5, bomber.pos.y + dy * 5)
                if self.is_valid_position(check_pos) and check_pos not in self.walls:
                    if self.check_exit_within_radius(check_pos, radius=5):
                        return False  # Нашли выход в этом направлении
        
        # Если не нашли выход для всех бомберов - пространство замкнуто
        # Дополнительная проверка: если все бомберы окружены препятствиями
        all_blocked = True
        for bomber in alive_bombers:
            blocked_count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                if not self.is_valid_position(neighbor) or neighbor in self.walls:
                    blocked_count += 1
                elif neighbor in self.obstacles and (self.booster_manager.acrobatics_level if self.booster_manager else 0) < 2:
                    blocked_count += 1
            if blocked_count < 3:  # Есть хотя бы один свободный выход
                all_blocked = False
                break
        
        # Если все бомберы заблокированы и нет выхода - пространство замкнуто
        return all_blocked or len(self.obstacles) > 0
    
    def find_enclosed_space_breakthrough_plan(self) -> Optional[Dict]:
        """
        Найти план размещения бомб для замкнутого пространства.
        
        Алгоритм:
        1. Определяет, что пространство замкнуто (нет выхода)
        2. Генерирует комбинации размещения до 6 бомб
        3. Проверяет, что у каждого бомбера есть безопасная клетка для отступления
        4. Выбирает лучший план по количеству разрушенных препятствий
        
        Returns:
            Dict с планом: {
                'bomb_placements': [(bomber_id, bomb_pos, escape_pos), ...],
                'obstacles_to_destroy': [Position, ...] - препятствия, которые будут разрушены
            }
        """
        alive_bombers = [b for b in self.bombers if b.alive and b.bombs_available > 0]
        if not alive_bombers:
            return None
        
        # Проверяем, что пространство действительно замкнуто
        if not self.is_space_truly_enclosed(alive_bombers):
            return None
        
        print(f"[ENCLOSED SPACE] Detected truly enclosed space with {len(alive_bombers)} bombers")
        
        # Находим все препятствия в радиусе досягаемости
        reachable_obstacles = []
        bomber_positions = {b.id: b.pos for b in alive_bombers}
        
        for bomber in alive_bombers:
            for obs in self.obstacles:
                dist = bomber.pos.manhattan_distance(obs)
                if 1 <= dist <= self.bomb_range + 5:  # Расширенный радиус
                    if obs not in reachable_obstacles:
                        reachable_obstacles.append(obs)
        
        if not reachable_obstacles:
            print(f"[ENCLOSED SPACE] No reachable obstacles found")
            return None
        
        print(f"[ENCLOSED SPACE] Found {len(reachable_obstacles)} reachable obstacles")
        
        # Генерируем возможные позиции для бомб
        bomb_position_candidates = []
        for obs in reachable_obstacles:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                bomb_pos = Position(obs.x + dx, obs.y + dy)
                
                if not self.is_valid_position(bomb_pos):
                    continue
                
                # Проверяем, попадет ли препятствие в радиус взрыва
                in_range = False
                if bomb_pos.x == obs.x:
                    y_dist = abs(bomb_pos.y - obs.y)
                    if y_dist <= self.bomb_range:
                        blocked = False
                        step = 1 if obs.y > bomb_pos.y else -1
                        for y in range(bomb_pos.y + step, obs.y, step):
                            if Position(bomb_pos.x, y) in self.walls:
                                blocked = True
                                break
                        if not blocked:
                            in_range = True
                elif bomb_pos.y == obs.y:
                    x_dist = abs(bomb_pos.x - obs.x)
                    if x_dist <= self.bomb_range:
                        blocked = False
                        step = 1 if obs.x > bomb_pos.x else -1
                        for x in range(bomb_pos.x + step, obs.x, step):
                            if Position(x, bomb_pos.y) in self.walls:
                                blocked = True
                                break
                        if not blocked:
                            in_range = True
                
                if in_range:
                    bomb_position_candidates.append((bomb_pos, obs))
        
        if not bomb_position_candidates:
            print(f"[ENCLOSED SPACE] No valid bomb positions found")
            return None
        
        print(f"[ENCLOSED SPACE] Generated {len(bomb_position_candidates)} bomb position candidates")
        
        # Пробуем найти план с разным количеством бомб (от 1 до min(6, количество бомберов))
        max_bombs = min(6, len(alive_bombers))
        best_plan = None
        best_score = -1
        
        # Используем жадный алгоритм: пробуем разместить бомбы по одной, проверяя безопасность
        used_bombers = set()
        planned_bombs = []
        bomb_placements = []
        
        # Сортируем кандидатов по приоритету (ближе к препятствиям, больше препятствий в радиусе)
        def score_bomb_candidate(bomb_pos, obs):
            score = 0
            # Приоритет: ближе к препятствию
            score += 10.0 / (bomb_pos.manhattan_distance(obs) + 1)
            # Бонус за количество препятствий в радиусе
            obstacles_in_range = 0
            for other_obs in reachable_obstacles:
                if bomb_pos.x == other_obs.x and abs(bomb_pos.y - other_obs.y) <= self.bomb_range:
                    obstacles_in_range += 1
                elif bomb_pos.y == other_obs.y and abs(bomb_pos.x - other_obs.x) <= self.bomb_range:
                    obstacles_in_range += 1
            score += obstacles_in_range * 5
            return score
        
        bomb_position_candidates.sort(key=lambda x: score_bomb_candidate(x[0], x[1]), reverse=True)
        
        # Пробуем разместить бомбы
        for bomb_pos, target_obs in bomb_position_candidates[:max_bombs * 3]:  # Проверяем больше кандидатов
            if len(planned_bombs) >= max_bombs:
                break
            
            # Находим ближайшего бомбера, который еще не использован
            best_bomber = None
            best_bomber_dist = float('inf')
            
            for bomber in alive_bombers:
                if bomber.id in used_bombers:
                    continue
                
                dist = bomber.pos.manhattan_distance(bomb_pos)
                if dist < best_bomber_dist:
                    best_bomber_dist = dist
                    best_bomber = bomber
            
            if not best_bomber:
                continue
            
            # Проверяем, не попадут ли другие бомберы под взрыв
            if self.will_bomb_hit_other_bombers(bomb_pos, best_bomber.id, planned_bombs=planned_bombs):
                continue
            
            # Проверяем, есть ли безопасная клетка для этого бомбера после всех взрывов
            all_bombs = planned_bombs + [bomb_pos]
            safe_escape = self.find_safe_escape_for_bomber(best_bomber, all_bombs)
            
            if not safe_escape:
                # Если не нашли безопасный путь, пробуем найти хотя бы одну безопасную соседнюю клетку
                # (более мягкие требования для замкнутого пространства)
                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    escape_candidate = Position(best_bomber.pos.x + edx, best_bomber.pos.y + edy)
                    if not self.is_valid_position(escape_candidate):
                        continue
                    
                    # Проверяем, что escape позиция не в радиусе взрыва
                    safe_from_bombs = True
                    for check_bomb in all_bombs:
                        if escape_candidate.x == check_bomb.x and abs(escape_candidate.y - check_bomb.y) <= self.bomb_range:
                            safe_from_bombs = False
                            break
                        elif escape_candidate.y == check_bomb.y and abs(escape_candidate.x - check_bomb.x) <= self.bomb_range:
                            safe_from_bombs = False
                            break
                    
                    if safe_from_bombs:
                        safe_escape = escape_candidate
                        break
                
                if not safe_escape:
                    continue  # Не нашли даже соседнюю безопасную клетку
            
            # Найден безопасный план для этой бомбы
            planned_bombs.append(bomb_pos)
            bomb_placements.append((best_bomber.id, bomb_pos, safe_escape))
            used_bombers.add(best_bomber.id)
            
            print(f"[ENCLOSED SPACE] Assigned bomb at {bomb_pos} to {best_bomber.id}, escape to {safe_escape}")
        
        if not bomb_placements:
            print(f"[ENCLOSED SPACE] Could not find safe plan")
            return None
        
        # Подсчитываем, сколько препятствий будет разрушено
        obstacles_to_destroy = set()
        for bomb_pos in planned_bombs:
            for obs in reachable_obstacles:
                # Проверяем, попадет ли препятствие в радиус взрыва
                if bomb_pos.x == obs.x:
                    y_dist = abs(bomb_pos.y - obs.y)
                    if y_dist <= self.bomb_range:
                        blocked = False
                        step = 1 if obs.y > bomb_pos.y else -1
                        for y in range(bomb_pos.y + step, obs.y, step):
                            if Position(bomb_pos.x, y) in self.walls:
                                blocked = True
                                break
                        if not blocked:
                            obstacles_to_destroy.add(obs)
                elif bomb_pos.y == obs.y:
                    x_dist = abs(bomb_pos.x - obs.x)
                    if x_dist <= self.bomb_range:
                        blocked = False
                        step = 1 if obs.x > bomb_pos.x else -1
                        for x in range(bomb_pos.x + step, obs.x, step):
                            if Position(x, bomb_pos.y) in self.walls:
                                blocked = True
                                break
                        if not blocked:
                            obstacles_to_destroy.add(obs)
        
        print(f"[ENCLOSED SPACE] Plan created: {len(bomb_placements)} bombs, {len(obstacles_to_destroy)} obstacles to destroy")
        
        return {
            'bomb_placements': bomb_placements,
            'obstacles_to_destroy': list(obstacles_to_destroy)
        }
    
    def find_safe_escape_for_bomber(self, bomber: Bomber, planned_bombs: List[Position]) -> Optional[Position]:
        """
        Найти безопасную клетку для бомбера после размещения всех планируемых бомб.
        
        Args:
            bomber: Бомбер, для которого ищем безопасную клетку
            planned_bombs: Список позиций планируемых бомб
            
        Returns:
            Безопасная позиция или None
        """
        bomb_timer = self.bomb_delay / 1000.0
        
        # Ищем безопасные позиции в радиусе
        search_radius = 15
        best_escape = None
        best_score = -1
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if abs(dx) + abs(dy) > search_radius:
                    continue
                
                escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                
                if not self.is_valid_position(escape_pos):
                    continue
                
                # Проверяем, что позиция не в радиусе взрыва ни одной из планируемых бомб
                safe = True
                min_dist_to_bomb = float('inf')
                
                for bomb_pos in planned_bombs:
                    dist_to_bomb = escape_pos.manhattan_distance(bomb_pos)
                    min_dist_to_bomb = min(min_dist_to_bomb, dist_to_bomb)
                    
                    # Проверяем, попадет ли позиция под взрыв
                    if escape_pos.x == bomb_pos.x:
                        y_dist = abs(escape_pos.y - bomb_pos.y)
                        if y_dist <= self.bomb_range:
                            blocked = False
                            step = 1 if escape_pos.y > bomb_pos.y else -1
                            for y in range(bomb_pos.y + step, escape_pos.y, step):
                                if Position(bomb_pos.x, y) in self.walls:
                                    blocked = True
                                    break
                            if not blocked:
                                safe = False
                                break
                    elif escape_pos.y == bomb_pos.y:
                        x_dist = abs(escape_pos.x - bomb_pos.x)
                        if x_dist <= self.bomb_range:
                            blocked = False
                            step = 1 if escape_pos.x > bomb_pos.x else -1
                            for x in range(bomb_pos.x + step, escape_pos.x, step):
                                if Position(x, bomb_pos.y) in self.walls:
                                    blocked = True
                                    break
                            if not blocked:
                                safe = False
                                break
                
                if not safe:
                    continue
                
                # Проверяем безопасность от других бомб (уже установленных)
                danger = self.get_blast_danger(escape_pos)
                if danger < bomb_timer + 0.5:  # Должна быть безопасна после всех взрывов
                    continue
                
                # Проверяем, есть ли путь к этой позиции
                path = self.find_path(bomber.pos, escape_pos, max_steps=20)
                if not path or len(path) < 2:
                    continue
                
                # Оцениваем позицию: дальше от бомб = лучше, ближе к бомберу = лучше
                score = min_dist_to_bomb * 2 - bomber.pos.manhattan_distance(escape_pos) + danger * 10
                
                if score > best_score:
                    best_score = score
                    best_escape = escape_pos
        
        return best_escape
    
    def find_coordinated_breakthrough_plan(self) -> Optional[Dict]:
        """Найти координированный план размещения бомб для открытия выходов
        
        Returns:
            Dict с планом: {
                'bomb_placements': [(bomber_id, bomb_pos, escape_pos), ...],
                'safe_zone': Position - безопасная зона для всех бомберов
            }
        """
        alive_bombers = [b for b in self.bombers if b.alive and b.bombs_available > 0]
        if len(alive_bombers) < 2:
            return None
        
        # Находим препятствия, которые блокируют выход
        blocking_obstacles = []
        for bomber in alive_bombers:
            for obs in self.obstacles:
                dist = bomber.pos.manhattan_distance(obs)
                if 1 <= dist <= self.bomb_range + 2:
                    # Проверяем, блокирует ли это препятствие выход
                    # Если препятствие на пути к свободному пространству
                    blocking_obstacles.append((obs, dist, bomber))
        
        if not blocking_obstacles:
            return None
        
        # Сортируем препятствия по приоритету (ближайшие и блокирующие)
        blocking_obstacles.sort(key=lambda x: x[1])
        
        bomb_timer = self.bomb_delay / 1000.0
        speed = 2 + (self.booster_manager.speed_level if self.booster_manager else 0)
        
        # Планируем размещение бомб
        bomb_placements = []
        used_bombers = set()
        planned_bomb_positions = []
        
        for obs, dist, bomber in blocking_obstacles[:min(6, len(alive_bombers))]:
            if bomber.id in used_bombers:
                continue
            
            # Ищем позицию для бомбы рядом с препятствием
            best_bomb_pos = None
            best_escape = None
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                bomb_pos = Position(obs.x + dx, obs.y + dy)
                if not self.is_valid_position(bomb_pos):
                    continue
                
                # Проверяем, попадет ли препятствие в радиус взрыва
                in_range = False
                if bomb_pos.x == obs.x and abs(bomb_pos.y - obs.y) <= self.bomb_range:
                    in_range = True
                elif bomb_pos.y == obs.y and abs(bomb_pos.x - obs.x) <= self.bomb_range:
                    in_range = True
                
                if not in_range:
                    continue
                
                # Проверяем, не попадут ли другие бомберы под взрыв
                if self.will_bomb_hit_other_bombers(bomb_pos, bomber.id, planned_bombs=planned_bomb_positions):
                    continue
                
                # Ищем escape позицию для этого бомбера
                escape_path = self.find_escape_path_after_bomb(bomber, bomb_pos)
                if escape_path:
                    best_bomb_pos = bomb_pos
                    best_escape = escape_path[-1] if escape_path else None
                    break
                else:
                    # Пробуем найти хотя бы одну безопасную клетку
                    for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                        if self.is_valid_position(escape_pos):
                            # Проверяем, что escape позиция не в радиусе взрыва
                            if escape_pos.x != bomb_pos.x or abs(escape_pos.y - bomb_pos.y) > self.bomb_range:
                                if escape_pos.y != bomb_pos.y or abs(escape_pos.x - bomb_pos.x) > self.bomb_range:
                                    # Проверяем безопасность от всех планируемых бомб
                                    all_bombs = planned_bomb_positions + [bomb_pos]
                                    safe = True
                                    for check_bomb in all_bombs:
                                        if escape_pos.x == check_bomb.x and abs(escape_pos.y - check_bomb.y) <= self.bomb_range:
                                            safe = False
                                            break
                                        elif escape_pos.y == check_bomb.y and abs(escape_pos.x - check_bomb.x) <= self.bomb_range:
                                            safe = False
                                            break
                                    if safe:
                                        best_bomb_pos = bomb_pos
                                        best_escape = escape_pos
                                        break
                        if best_escape:
                            break
            
            if best_bomb_pos and best_escape:
                bomb_placements.append((bomber.id, best_bomb_pos, best_escape))
                planned_bomb_positions.append(best_bomb_pos)
                used_bombers.add(bomber.id)
        
        if not bomb_placements:
            return None
        
        # Находим безопасную зону для всех бомберов после взрывов
        # Ищем позицию, которая не попадет под взрывы всех планируемых бомб
        safe_zone = None
        all_bomber_positions = {b.pos for b in alive_bombers}
        center_pos = alive_bombers[0].pos
        
        # Ищем позицию на расстоянии от всех бомб (расширяем радиус поиска)
        best_safe_zone = None
        best_score = -1
        
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                if abs(dx) + abs(dy) > 15:
                    continue
                check_pos = Position(center_pos.x + dx, center_pos.y + dy)
                if not self.is_valid_position(check_pos):
                    continue
                
                # Проверяем, что позиция не в радиусе взрыва ни одной из планируемых бомб
                safe = True
                min_dist_to_bomb = float('inf')
                for bomb_pos in planned_bomb_positions:
                    # Проверяем расстояние до бомбы
                    dist_to_bomb = check_pos.manhattan_distance(bomb_pos)
                    min_dist_to_bomb = min(min_dist_to_bomb, dist_to_bomb)
                    
                    # Проверяем, попадет ли позиция под взрыв
                    if check_pos.x == bomb_pos.x and abs(check_pos.y - bomb_pos.y) <= self.bomb_range:
                        safe = False
                        break
                    elif check_pos.y == bomb_pos.y and abs(check_pos.x - bomb_pos.x) <= self.bomb_range:
                        safe = False
                        break
                
                if safe and min_dist_to_bomb > self.bomb_range:
                    # Проверяем безопасность от других бомб (уже установленных)
                    danger = self.get_blast_danger(check_pos)
                    if danger > bomb_timer + 1.0:  # Безопасна после всех взрывов
                        # Оцениваем позицию: дальше от бомб = лучше
                        score = min_dist_to_bomb + danger * 10
                        if score > best_score:
                            best_score = score
                            best_safe_zone = check_pos
        
        safe_zone = best_safe_zone
        
        if safe_zone:
            return {
                'bomb_placements': bomb_placements,
                'safe_zone': safe_zone
            }
        
        return None

    def get_safe_zone_escape_path(self, bomber: Bomber, max_steps: int = 30) -> Optional[List[Position]]:
        """Найти путь до ближайшей безопасной зоны из анализа карты"""
        if not (self.use_local_api and self.api):
            return None

        analysis = self.api.get_map_analysis(force_update=False)
        if not analysis or not analysis.safe_zones:
            return None

        safe_positions = sorted(
            analysis.safe_zones,
            key=lambda pos: bomber.pos.manhattan_distance(Position(pos[0], pos[1]))
        )

        for pos in safe_positions[:40]:
            safe_target = Position(pos[0], pos[1])
            if not self.is_valid_position(safe_target):
                continue
            path = self.find_path(bomber.pos, safe_target, max_steps=max_steps)
            if path:
                print(f"[ESCAPE] {bomber.id} targeting safe zone at {safe_target}")
                return path

        return None

    def attempt_enclosure_escape(self, bomber: Bomber, my_commands: Dict, commands: List[Dict]) -> bool:
        """Пытаемся выйти из замкнутого пространства - агрессивный режим"""
        print(f"[ENCLOSED] {bomber.id} is enclosed, attempting escape...")
        
        # ПРИОРИТЕТ 0: Атакуем врагов, если они рядом (даже когда заперт)
        if bomber.bombs_available > 0:
            # Ищем врагов в радиусе взрыва
            nearby_enemy = self.find_nearby_enemy(bomber, max_distance=self.bomb_range + 2)
            if nearby_enemy:
                enemy_dist = bomber.pos.manhattan_distance(nearby_enemy.pos)
                print(f"[ENCLOSED] {bomber.id} sees enemy at {nearby_enemy.pos}, distance={enemy_dist}")
                
                # Пробуем найти позицию для бомбы, чтобы убить врага
                bomb_result = self.find_bomb_position_for_enemy(bomber, nearby_enemy)
                if bomb_result:
                    bomb_pos, escape_path = bomb_result
                    dist_to_bomb = bomber.pos.manhattan_distance(bomb_pos)
                    if bomber.pos == bomb_pos:
                        # Проверяем, не попадут ли другие юниты под взрыв
                        if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                            # Бомба ставится на текущей позиции
                            my_commands["path"] = [[p.x, p.y] for p in escape_path[:10]]
                            my_commands["bombs"] = [[bomb_pos.x, bomb_pos.y]]
                            commands.append(my_commands)
                            print(f"[ENCLOSED] {bomber.id} attacking enemy with bomb at {bomb_pos} (current pos)")
                            return True
                    else:
                        # Идем к позиции бомбы, бомба ставится на последней позиции пути
                        path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=4)
                        if path_to_bomb and len(path_to_bomb) > 1:
                            bomb_placement_pos = path_to_bomb[-1]
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                commands.append(my_commands)
                                print(f"[ENCLOSED] {bomber.id} moving to attack enemy at {bomb_pos}")
                                return True
                
                # Если враг очень близко, пробуем поставить бомбу прямо рядом с ним
                if enemy_dist <= self.bomb_range + 1:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        bomb_pos = Position(nearby_enemy.pos.x + dx, nearby_enemy.pos.y + dy)
                        if not self.is_valid_position(bomb_pos):
                            continue
                        
                        # Проверяем, попадет ли враг в радиус взрыва
                        in_range = False
                        if bomb_pos.x == nearby_enemy.pos.x and abs(bomb_pos.y - nearby_enemy.pos.y) <= self.bomb_range:
                            in_range = True
                        elif bomb_pos.y == nearby_enemy.pos.y and abs(bomb_pos.x - nearby_enemy.pos.x) <= self.bomb_range:
                            in_range = True
                        
                        if in_range:
                            # Пробуем поставить бомбу (агрессивный режим)
                            if bomber.pos == bomb_pos:
                                # Бомба ставится на текущей позиции
                                # Ищем escape позицию
                                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                                    if self.is_valid_position(escape_pos):
                                        if escape_pos.manhattan_distance(bomb_pos) >= self.bomb_range + 1:
                                            if self.get_blast_danger(escape_pos) > 0.5:  # Более мягкие требования
                                                # Проверяем, не попадут ли другие юниты под взрыв
                                                if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                                                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                                                    my_commands["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                    commands.append(my_commands)
                                                    print(f"[ENCLOSED] {bomber.id} attacking nearby enemy with bomb at {bomber.pos} (current)")
                                                    return True
        
        # ПРИОРИТЕТ 1: Если есть бомбы, агрессивно ищем препятствия для уничтожения
        if bomber.bombs_available > 0:
            # Используем агрессивный режим поиска
            breakthrough = self.find_bomb_position_for_obstacle_breakthrough(bomber, aggressive=True)
            if breakthrough:
                bomb_pos, escape_path = breakthrough
                print(f"[ENCLOSED] {bomber.id} found breakthrough at {bomb_pos}")
                
                # Если мы уже на позиции - ставим бомбу
                if bomber.pos == bomb_pos:
                    # Проверяем, не попадут ли другие юниты под взрыв
                    if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                        # Бомба ставится на текущей позиции
                        my_commands["path"] = [[p.x, p.y] for p in escape_path[:10]]
                        my_commands["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                        commands.append(my_commands)
                        print(f"[ENCLOSED] {bomber.id} placing bomb at {bomber.pos} (current) and escaping")
                        return True
                    else:
                        print(f"[ENCLOSED] {bomber.id} cannot place bomb at {bomb_pos} - would hit other bombers")
                else:
                    # Идем к позиции бомбы, бомба ставится на последней позиции пути
                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=5)
                    if path_to_bomb and len(path_to_bomb) > 1:
                        bomb_placement_pos = path_to_bomb[-1]
                        # Проверяем, не попадут ли другие юниты под взрыв
                        if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                            my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                            my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            commands.append(my_commands)
                            print(f"[ENCLOSED] {bomber.id} moving to bomb position {bomb_pos}")
                            return True
                        else:
                            print(f"[ENCLOSED] {bomber.id} cannot place bomb at {bomb_placement_pos} - would hit other bombers")
        
        # ПРИОРИТЕТ 2: Ищем ближайшие препятствия для атаки (даже если нет прямого пути)
        if bomber.bombs_available > 0:
            # Ищем препятствия в радиусе взрыва
            for obs in self.obstacles:
                dist = bomber.pos.manhattan_distance(obs)
                if dist <= self.bomb_range + 2:  # В радиусе взрыва + небольшой запас
                    # Проверяем, можем ли мы поставить бомбу так, чтобы она попала в препятствие
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                        bomb_pos = Position(obs.x + dx, obs.y + dy)
                        
                        if bomb_pos == obs and (self.booster_manager.acrobatics_level if self.booster_manager else 0) < 2:
                            continue
                            
                        if not self.is_valid_position(bomb_pos):
                            continue
                        
                        # Проверяем, попадет ли препятствие в радиус взрыва
                        in_range = False
                        if bomb_pos.x == obs.x and abs(bomb_pos.y - obs.y) <= self.bomb_range:
                            in_range = True
                        elif bomb_pos.y == obs.y and abs(bomb_pos.x - obs.x) <= self.bomb_range:
                            in_range = True
                        
                        if in_range:
                            # Пробуем найти путь к позиции бомбы или поставить её прямо здесь
                            if bomber.pos.manhattan_distance(bomb_pos) <= 1:
                                # Можем поставить бомбу прямо здесь
                                # Ищем escape позицию
                                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                                    if self.is_valid_position(escape_pos):
                                        if escape_pos.manhattan_distance(bomb_pos) >= self.bomb_range + 1:
                                            if self.get_blast_danger(escape_pos) > 1.0:
                                                # Проверяем, не попадут ли другие юниты под взрыв
                                                if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                                                    # Бомба ставится на текущей позиции
                                                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                                                    my_commands["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                    commands.append(my_commands)
                                                    print(f"[ENCLOSED] {bomber.id} placing bomb at {bomber.pos} (current) to destroy obstacle at {obs}")
                                                    return True
                            else:
                                # Пробуем найти путь к позиции бомбы
                                path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=5)
                                if path_to_bomb and len(path_to_bomb) > 1:
                                    # Проверяем, не попадут ли другие юниты под взрыв
                                    bomb_placement_pos = path_to_bomb[-1]
                                    if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                        # Бомба ставится на последней позиции пути
                                        my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                        my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                        commands.append(my_commands)
                                        print(f"[ENCLOSED] {bomber.id} moving to bomb position {bomb_pos} to destroy obstacle")
                                        return True
        
        # ПРИОРИТЕТ 3: Пробуем найти путь к безопасной зоне
        escape_path = self.get_safe_zone_escape_path(bomber)
        if escape_path:
            my_commands["path"] = [[p.x, p.y] for p in escape_path[:12]]
            commands.append(my_commands)
            print(f"[ENCLOSED] {bomber.id} moving to safe zone")
            return True

        # ПРИОРИТЕТ 4: Если ничего не помогло, пробуем просто двигаться в любую доступную сторону
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            move_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
            if self.is_valid_position(move_pos):
                # Проверяем, что это не препятствие (или есть акробатика)
                if move_pos not in self.obstacles or (self.booster_manager.acrobatics_level if self.booster_manager else 0) >= 2:
                    danger = self.get_blast_danger(move_pos)
                    if danger > 0.5:  # Хотя бы минимальная безопасность
                        my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [move_pos.x, move_pos.y]]
                        commands.append(my_commands)
                        print(f"[ENCLOSED] {bomber.id} making emergency move to {move_pos}")
                        return True

        return False
    
    def buy_booster(self, booster_type: int, retry_count: int = 2) -> Dict:
        """Купить улучшение"""
        if self.use_local_api and self.api:
            return self.api.buy_booster(booster_type)

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                payload = {"booster": booster_type}
                response = self.session.post(f"{self.base_url}/api/booster", json=payload)
                if response.status_code == 429:
                    # Устанавливаем общий backoff на 5 секунд
                    self.backoff_until = time.time() + 5.0
                    sleep_time = 5.0
                    print(f"[RATE_LIMIT] Got 429 in buy_booster, setting backoff for {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                response.raise_for_status()
                # Сбрасываем backoff при успехе
                self.backoff_until = 0
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(2.0)
    
    def update_state(self):
        """Обновить состояние игры"""
        data = self.get_arena()

        if "map_size" in data:
            self.map_size = tuple(data["map_size"])

        self.walls = {Position(x, y) for x, y in data.get("arena", {}).get("walls", [])}
        self.obstacles = {Position(x, y) for x, y in data.get("arena", {}).get("obstacles", [])}

        self.bombs = []
        for bomb_data in data.get("arena", {}).get("bombs", []):
            pos = Position(bomb_data["pos"][0], bomb_data["pos"][1])
            self.bombs.append(Bomb(pos, bomb_data["timer"], bomb_data["range"]))

        self.bombers = []
        for bomber_data in data.get("bombers", []):
            pos = Position(bomber_data["pos"][0], bomber_data["pos"][1])
            self.bombers.append(Bomber(
                bomber_data["id"],
                pos,
                bomber_data["alive"],
                bomber_data["can_move"],
                bomber_data["bombs_available"],
                bomber_data.get("armor", 0),
                bomber_data.get("safe_time", 0)
            ))

        self.enemies = []
        for enemy_data in data.get("enemies", []):
            pos = Position(enemy_data["pos"][0], enemy_data["pos"][1])
            self.enemies.append(Enemy(
                enemy_data["id"],
                pos,
                enemy_data.get("safe_time", 0)
            ))

        self.mobs = []
        active_ghosts = 0
        for mob_data in data.get("mobs", []):
            pos = Position(mob_data["pos"][0], mob_data["pos"][1])
            mob = Mob(
                mob_data["id"],
                pos,
                mob_data.get("type", "patrol"),
                int(mob_data.get("safe_time", 0))
            )
            self.mobs.append(mob)
            if mob.type == "ghost" and mob.safe_time == 0:
                active_ghosts += 1
                print(f"[GHOST] Active ghost at {pos} (id: {mob.id})")

        if active_ghosts > 0:
            print(f"[WARNING] {active_ghosts} active ghosts detected!")

        # Отслеживаем изменения для анализа начисления очков
        current_enemy_count = len(self.enemies)
        current_mob_count = len(self.mobs)
        current_obstacle_count = len(self.obstacles)

        # Анализируем изменения
        enemy_diff = self.last_enemy_count - current_enemy_count
        mob_diff = self.last_mob_count - current_mob_count
        obstacle_diff = self.last_obstacle_count - current_obstacle_count

        if enemy_diff > 0:
            self.total_kills += enemy_diff
            print(f"[KILL] Enemies killed: +{enemy_diff} (total: {self.total_kills})")

        if mob_diff > 0:
            self.total_kills += mob_diff
            print(f"[KILL] Mobs killed: +{mob_diff} (total: {self.total_kills})")

        if obstacle_diff > 0:
            self.total_obstacles_destroyed += obstacle_diff
            print(f"[DESTROY] Obstacles destroyed: +{obstacle_diff} (total: {self.total_obstacles_destroyed})")

        # Обновляем предыдущие значения
        self.last_enemy_count = current_enemy_count
        self.last_mob_count = current_mob_count
        self.last_obstacle_count = current_obstacle_count
        
        # Инициализируем продвинутую логику, если включена
        if self.use_advanced_logic and self.map_size:
            self._update_advanced_logic()
        
        # Инициализируем модули после обновления состояния
        self._init_modules()
    
    def _init_modules(self):
        """Инициализировать модули после обновления состояния"""
        if not self.map_size:
            return
        
        # Инициализируем PathFinder
        if self.path_finder is None:
            self.path_finder = PathFinder(
                self.map_size, self.walls, self.obstacles, self.bombs,
                self.mobs, self.enemies, self.bomb_range, self.bomb_delay,
                self.booster_manager.speed_level if self.booster_manager else 0,
                self.booster_manager.acrobatics_level if self.booster_manager else 0,
                self.safe_pathfinder, self.use_advanced_logic
            )
        else:
            # Обновляем состояние PathFinder
            self.path_finder.map_size = self.map_size
            self.path_finder.walls = self.walls
            self.path_finder.obstacles = self.obstacles
            self.path_finder.bombs = self.bombs
            self.path_finder.mobs = self.mobs
            self.path_finder.enemies = self.enemies
            self.path_finder.bomb_range = self.bomb_range
            self.path_finder.bomb_delay = self.bomb_delay
            if self.booster_manager:
                self.path_finder.speed_level = self.booster_manager.speed_level
                self.path_finder.acrobatics_level = self.booster_manager.acrobatics_level
            self.path_finder.safe_pathfinder = self.safe_pathfinder
        
        # Инициализируем BoosterManager
        if self.booster_manager is None:
            self.booster_manager = BoosterManager(
                self.get_boosters,
                self.buy_booster
            )
        
        # Инициализируем BomberCoordinator
        if self.coordinator is None:
            self.coordinator = BomberCoordinator(
                self.check_exit_within_radius,
                self.get_collective_targets
            )
        
        # Инициализируем CombatManager
        if self.combat_manager is None:
            self.combat_manager = CombatManager(self)
        
        # Инициализируем GameStrategy
        if self.strategy is None:
            self.strategy = GameStrategy(self)
    
    def _update_advanced_logic(self):
        """Обновить продвинутую логику (DeathMap, TargetSelector, SafePathfinder)"""
        if not self.map_size:
            return
        
        # Конвертируем Position в tuple для advanced_logic
        walls_set = {(p.x, p.y) for p in self.walls}
        obstacles_set = {(p.x, p.y) for p in self.obstacles}
        
        # Инициализируем MapPredictor, если еще не создан
        if self.map_predictor is None:
            self.map_predictor = MapPredictor(
                self.map_size, walls_set, obstacles_set, self.bomb_delay
            )
        else:
            # Обновляем стены и препятствия
            self.map_predictor.walls = walls_set
            self.map_predictor.obstacles = obstacles_set
            self.map_predictor.bomb_delay_ms = self.bomb_delay
        
        # Конвертируем бомбы в формат для симуляции
        bombs_data = []
        for bomb in self.bombs:
            bombs_data.append({
                'x': bomb.pos.x,
                'y': bomb.pos.y,
                'timer': bomb.timer * 1000 if bomb.timer < 100 else bomb.timer,  # Конвертируем в мс
                'range': bomb.range
            })
        
        # Симулируем цепные реакции и создаем DeathMap
        time_bombs = self.map_predictor.simulate_chain_reactions(bombs_data)
        self.map_predictor.build_death_map(time_bombs)
        
        # Инициализируем TargetSelector
        if self.target_selector is None:
            self.target_selector = TargetSelector(obstacles_set, self.bomb_range)
        else:
            self.target_selector.obstacles = obstacles_set
            self.target_selector.bomb_range = self.bomb_range
        
        # Инициализируем SafePathfinder
        enemies_data = [{'x': e.pos.x, 'y': e.pos.y} for e in self.enemies]
        mobs_data = [{'x': m.pos.x, 'y': m.pos.y, 'type': m.type} for m in self.mobs]
        
        speed_level = self.booster_manager.speed_level if self.booster_manager else 0
        if self.safe_pathfinder is None:
            self.safe_pathfinder = SafePathfinder(
                self.map_predictor, walls_set, obstacles_set,
                enemies_data, mobs_data, speed=2 + speed_level
            )
        else:
            self.safe_pathfinder.predictor = self.map_predictor
            self.safe_pathfinder.walls = walls_set
            self.safe_pathfinder.obstacles = obstacles_set
            self.safe_pathfinder.enemies = enemies_data
            self.safe_pathfinder.mobs = mobs_data
            self.safe_pathfinder.speed = 2 + speed_level
    
    def is_valid_position(self, pos: Position) -> bool:
        """Проверить, находится ли позиция в пределах карты и не является ли стеной"""
        if not self.map_size: return False
        if not (0 <= pos.x < self.map_size[0] and 0 <= pos.y < self.map_size[1]):
            return False
        if pos in self.walls:
            return False
        # Препятствия блокируют движение, если нет акробатики
        if (self.booster_manager.acrobatics_level if self.booster_manager else 0) < 2 and pos in self.obstacles:
            return False
        return True
    
    def get_blast_danger(self, pos: Position, time_offset: float = 0.0) -> float:
        """
        Оценить опасность взрыва в данной точке.
        Возвращает время до взрыва (0 если прямо сейчас), или float('inf') если безопасно.
        Проверяет ВСЕ бомбы на карте (не только свои).
        
        Использует продвинутую логику с цепными реакциями, если доступна.
        """
        # Если доступна продвинутая логика, используем её
        if self.use_advanced_logic and self.map_predictor:
            arrival_tick = int(time_offset * 1000 / self.map_predictor.tick_duration_ms)
            danger_tick = self.map_predictor.get_danger_at_time(pos.x, pos.y, arrival_tick)
            if danger_tick is not None:
                # Конвертируем тик обратно в секунды
                danger_time_sec = (danger_tick - arrival_tick) * self.map_predictor.tick_duration_ms / 1000.0
                return max(0.0, danger_time_sec)
            return float('inf')
        
        # Fallback на старую логику
        min_timer = float('inf')
        is_in_danger = False
        
        for bomb in self.bombs:
            # Проверяем все бомбы, даже те, что взорвутся позже (для планирования)
            # Но приоритет - бомбы, которые взорвутся скоро
            if bomb.timer > 15.0: continue  # Слишком далеко в будущем
            
            # Проверяем попадание в радиус
            in_range = False
            if pos.y == bomb.pos.y and abs(pos.x - bomb.pos.x) <= bomb.range:
                # Проверка стен между бомбой и целью
                blocked = False
                step = 1 if pos.x > bomb.pos.x else -1
                for x in range(bomb.pos.x + step, pos.x + step, step): # range исключает конец, но нам надо проверить саму точку? Нет, луч
                    # Проверяем только промежуточные стены
                    if x == pos.x: break
                    if Position(x, pos.y) in self.walls:
                        blocked = True
                        break
                    # Препятствия тоже блокируют взрыв
                    if Position(x, pos.y) in self.obstacles:
                        blocked = True
                        break
                if not blocked: in_range = True
                
            elif pos.x == bomb.pos.x and abs(pos.y - bomb.pos.y) <= bomb.range:
                blocked = False
                step = 1 if pos.y > bomb.pos.y else -1
                for y in range(bomb.pos.y + step, pos.y + step, step):
                    if y == pos.y: break
                    if Position(pos.x, y) in self.walls:
                        blocked = True
                        break
                    if Position(pos.x, y) in self.obstacles:
                        blocked = True
                        break
                if not blocked: in_range = True
            
            if in_range:
                # Время до взрыва с учетом time_offset
                time_to_explosion = bomb.timer - time_offset
                if time_to_explosion <= 0:
                    # УЖЕ взрывается или взорвалась
                    return 0.0
                if time_to_explosion < min_timer:
                    min_timer = time_to_explosion
                    is_in_danger = True
                    
        return min_timer if is_in_danger else float('inf')
    
    def find_path(self, start: Position, target: Position, max_steps: int = 20) -> Optional[List[Position]]:
        """
        A* поиск пути с учетом опасности от ВСЕХ бомб.
        Использует продвинутую логику Space-Time A*, если доступна.
        """
        if self.path_finder:
            return self.path_finder.find_path(start, target, max_steps)
        # Fallback для случая, когда path_finder еще не инициализирован
        if start == target:
            return [target]
        return None
    
    def find_nearby_ghost(self, bomber: Bomber, max_distance: int = 12) -> Optional[Mob]:
        """Найти ближайшего призрака (ghost) в радиусе обзора"""
        nearest_ghost = None
        min_dist = float('inf')
        
        for mob in self.mobs:
            if mob.safe_time > 0: continue  # Игнорируем спящих
            if mob.type != "ghost": continue  # Только призраки
            
            dist = bomber.pos.manhattan_distance(mob.pos)
            # Призраки имеют радиус обзора 10, но мы проверяем до 12 для безопасности
            if dist <= max_distance and dist < min_dist:
                min_dist = dist
                nearest_ghost = mob
        
        return nearest_ghost
    
    def find_nearby_enemy(self, bomber: Bomber, max_distance: int = 3) -> Optional[Enemy]:
        """Найти ближайшего врага или моба"""
        nearest = None
        min_dist = float('inf')
        
        # Проверяем врагов
        for enemy in self.enemies:
            dist = bomber.pos.manhattan_distance(enemy.pos)
            if dist <= max_distance and dist < min_dist:
                # Проверяем, что это не наш бомбер
                is_our_bomber = any(b.id == enemy.id for b in self.bombers)
                if not is_our_bomber:
                    min_dist = dist
                    nearest = enemy
                    print(f"[FIND_ENEMY] {bomber.id} found enemy {enemy.id} at {enemy.pos}, distance={dist}")
                else:
                    print(f"[FIND_ENEMY] {bomber.id} enemy {enemy.id} is our own bomber, skipping")
        
        # Проверяем мобов (опасны при контакте)
        # ПРИОРИТЕТ: призраки опаснее, т.к. могут проходить через препятствия
        for mob in self.mobs:
            if mob.safe_time > 0: continue  # Игнорируем спящих
            dist = bomber.pos.manhattan_distance(mob.pos)
            if dist <= max_distance:
                # Призраки опаснее - они могут проходить через препятствия
                priority_dist = dist
                if mob.type == "ghost":
                    # Призраки опаснее даже на большем расстоянии
                    priority_dist = dist - 2  # Бонус приоритета
                
                if priority_dist < min_dist:
                    min_dist = priority_dist
                    # Создаем "псевдо-врага" из моба
                    nearest = Enemy(mob.id, mob.pos, mob.safe_time)
                    print(f"[FIND_ENEMY] {bomber.id} found mob {mob.id} ({mob.type}) at {mob.pos}, distance={dist}")
        
        if nearest is None and len(self.enemies) > 0:
            print(f"[FIND_ENEMY] {bomber.id} no enemies found within {max_distance} cells (enemies={len(self.enemies)}, closest_dist={min([bomber.pos.manhattan_distance(e.pos) for e in self.enemies]) if self.enemies else 'N/A'})")
        
        return nearest
    
    def check_exit_within_radius(self, pos: Position, radius: int = 5) -> bool:
        """Проверить, есть ли выход (свободная клетка без стен) в радиусе"""
        if self.path_finder:
            return self.path_finder.check_exit_within_radius(pos, radius)
        # Fallback
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                check_pos = Position(pos.x + dx, pos.y + dy)
                if not self.is_valid_position(check_pos):
                    continue
                acrobatics_level = self.booster_manager.acrobatics_level if self.booster_manager else 0
                if check_pos not in self.walls and (check_pos not in self.obstacles or acrobatics_level >= 2):
                    walls_around = 0
                    for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = Position(check_pos.x + ndx, check_pos.y + ndy)
                        if neighbor in self.walls:
                            walls_around += 1
                    if walls_around < 3:
                        return True
        return False
    
    def get_collective_targets(self, bombers: List[Bomber]) -> List[Position]:
        """Получить коллективные цели для группы бомберов - все ломающиеся стены"""
        targets = []
        # Собираем все препятствия, которые видны хотя бы одному бомберу
        for bomber in bombers:
            if not bomber.alive:
                continue
            for obs in self.obstacles:
                dist = bomber.pos.manhattan_distance(obs)
                if dist <= 15:  # В пределах видимости
                    if obs not in targets:
                        targets.append(obs)
        return targets
    
    def are_bombers_at_same_point(self, bombers: List[Bomber], tolerance: int = 1) -> bool:
        """Проверить, находятся ли все бомберы в одной точке (с допуском)"""
        if self.coordinator:
            return self.coordinator.are_bombers_at_same_point(bombers, tolerance)
        # Fallback
        if not bombers:
            return False
        alive_bombers = [b for b in bombers if b.alive]
        if len(alive_bombers) < 2:
            return False
        first_pos = alive_bombers[0].pos
        for bomber in alive_bombers[1:]:
            if bomber.pos.manhattan_distance(first_pos) > tolerance:
                return False
        return True
    
    def get_bomber_groups(self, max_distance: int = 3) -> List[List[Bomber]]:
        """Разделить бомберов на группы по близости"""
        if self.coordinator:
            return self.coordinator.get_bomber_groups(self.bombers, max_distance)
        # Fallback
        alive_bombers = [b for b in self.bombers if b.alive]
        if not alive_bombers:
            return []
        groups = []
        used = set()
        for bomber in alive_bombers:
            if bomber.id in used:
                continue
            group = [bomber]
            used.add(bomber.id)
            for other in alive_bombers:
                if other.id in used:
                    continue
                if bomber.pos.manhattan_distance(other.pos) <= max_distance:
                    group.append(other)
                    used.add(other.id)
            groups.append(group)
        return groups
    
    def find_safe_escape_from_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[List[Position]]:
        """Найти безопасный путь отступления от призрака"""
        if self.path_finder:
            return self.path_finder.find_safe_escape_from_ghost(bomber, ghost)
        return None
    
    def create_circular_path_around_target(self, bomber: Bomber, target_pos: Position, start_pos: Position) -> Optional[List[Position]]:
        """Создать круговой путь вокруг цели для уклонения
        
        Args:
            bomber: Бомбер
            target_pos: Позиция цели (моба/врага)
            start_pos: Начальная позиция для кругового движения
            
        Returns:
            Путь по кругу вокруг цели
        """
        # Создаем путь по кругу вокруг цели
        # Пробуем двигаться по часовой стрелке вокруг цели
        circular_path = [start_pos]
        
        # Находим направление от цели к стартовой позиции
        dx = start_pos.x - target_pos.x
        dy = start_pos.y - target_pos.y
        
        # Нормализуем направление
        if dx != 0:
            dx = 1 if dx > 0 else -1
        if dy != 0:
            dy = 1 if dy > 0 else -1
        
        # Создаем круговой путь: двигаемся перпендикулярно к направлению от цели
        # Пробуем 4 направления по кругу (по часовой стрелке)
        circle_directions = []
        if dx == 0:  # Движение по Y, крутимся по X
            if dy > 0:  # Ниже цели - идем вправо, вверх, влево
                circle_directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
            else:  # Выше цели - идем влево, вниз, вправо
                circle_directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        elif dy == 0:  # Движение по X, крутимся по Y
            if dx > 0:  # Справа от цели - идем вверх, влево, вниз
                circle_directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
            else:  # Слева от цели - идем вниз, вправо, вверх
                circle_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # Диагональ - используем перпендикулярные направления
            if dx > 0 and dy > 0:  # Справа-вниз
                circle_directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
            elif dx > 0 and dy < 0:  # Справа-вверх
                circle_directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
            elif dx < 0 and dy > 0:  # Слева-вниз
                circle_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            else:  # Слева-вверх
                circle_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        current_pos = start_pos
        for circle_dx, circle_dy in circle_directions[:6]:  # Берем первые 6 шагов для более длинного круга
            next_pos = Position(current_pos.x + circle_dx, current_pos.y + circle_dy)
            if self.is_valid_position(next_pos):
                # Проверяем, что позиция не слишком далеко от цели (остаемся рядом для следующей бомбы)
                dist_to_target = next_pos.manhattan_distance(target_pos)
                if dist_to_target <= self.bomb_range + 4:  # Остаемся в радиусе для следующей бомбы
                    # Проверяем безопасность от бомб
                    danger = self.get_blast_danger(next_pos)
                    if danger > 1.0:  # Безопасна от других бомб
                        circular_path.append(next_pos)
                        current_pos = next_pos
                    else:
                        # Пробуем следующее направление
                        continue
                else:
                    # Слишком далеко - пробуем следующее направление
                    continue
            else:
                # Невалидная позиция - пробуем следующее направление
                continue
        
        return circular_path if len(circular_path) > 1 else None
    
    def find_bomb_position_for_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить призрака"""
        if self.combat_manager:
            return self.combat_manager.find_bomb_position_for_ghost(bomber, ghost)
        return None
        
    def will_bomb_hit_other_bombers(self, bomb_pos: Position, bomber_id: str, planned_bombs: Optional[List[Position]] = None, planned_paths: Optional[Dict[str, List[Position]]] = None) -> bool:
        """Проверить, попадут ли другие юниты под взрыв бомбы"""
        if self.combat_manager:
            return self.combat_manager.will_bomb_hit_other_bombers(bomb_pos, bomber_id, planned_bombs, planned_paths)
        return False
    
    def can_safely_place_bomb(self, bomber: Bomber, bomb_pos: Position, aggressive: bool = False) -> Tuple[bool, Optional[List[Position]]]:
        """Проверить, можно ли безопасно поставить бомбу и убежать"""
        if self.combat_manager:
            return self.combat_manager.can_safely_place_bomb(bomber, bomb_pos, aggressive)
        return False, None
    
    def find_escape_path_after_bomb(self, bomber: Bomber, bomb_pos: Position) -> Optional[List[Position]]:
        """Найти путь отхода после установки бомбы"""
        if self.path_finder:
            return self.path_finder.find_escape_path_after_bomb(bomber, bomb_pos)
        return None
    
    def find_bomb_position_for_enemy(self, bomber: Bomber, enemy: Enemy) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить врага, и путь отступления"""
        if self.combat_manager:
            return self.combat_manager.find_bomb_position_for_enemy(bomber, enemy)
        return None
    
    def find_bomb_position_for_obstacle_breakthrough(self, bomber: Bomber, aggressive: bool = False) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы прорваться через препятствия
        
        Args:
            bomber: Юнит, который ищет прорыв
            aggressive: Если True, более агрессивный поиск (для запертых юнитов)
        """
        if bomber.bombs_available == 0:
            return None
        
        # Для запертых юнитов расширяем радиус поиска
        max_distance = 5 if aggressive else 3
        
        # Ищем препятствия рядом с юнитом
        nearby_obstacles = []
        for obs in self.obstacles:
            dist = bomber.pos.manhattan_distance(obs)
            if 1 <= dist <= max_distance:
                nearby_obstacles.append((obs, dist))
        
        if not nearby_obstacles:
            return None
    
        # Сортируем по близости
        nearby_obstacles.sort(key=lambda x: x[1])
        
        # Для агрессивного режима проверяем больше препятствий
        check_count = 10 if aggressive else 3
        
        # Пробуем найти позицию для бомбы рядом с ближайшими препятствиями
        for obs, dist in nearby_obstacles[:check_count]:
            # Проверяем все соседние позиции вокруг препятствия
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:  # Добавили (0,0) для случая, когда можем поставить бомбу прямо на препятствие
                bomb_pos = Position(obs.x + dx, obs.y + dy)
                
                # Пропускаем саму позицию препятствия (если нет акробатики)
                if bomb_pos == obs and (self.booster_manager.acrobatics_level if self.booster_manager else 0) < 2:
                    continue
                    
                if not self.is_valid_position(bomb_pos):
                    continue
                
                # Проверяем, что препятствие попадет в радиус взрыва
                # Взрыв идет по кресту, поэтому проверяем расстояние по осям
                in_range = False
                if bomb_pos.x == obs.x and abs(bomb_pos.y - obs.y) <= self.bomb_range:
                    # Проверяем, нет ли препятствий между бомбой и целью
                    blocked = False
                    step = 1 if obs.y > bomb_pos.y else -1
                    for y in range(bomb_pos.y + step, obs.y, step):
                        if Position(bomb_pos.x, y) in self.walls:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                elif bomb_pos.y == obs.y and abs(bomb_pos.x - obs.x) <= self.bomb_range:
                    blocked = False
                    step = 1 if obs.x > bomb_pos.x else -1
                    for x in range(bomb_pos.x + step, obs.x, step):
                        if Position(x, bomb_pos.y) in self.walls:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                
                if not in_range:
                    continue
                
                # Для агрессивного режима используем более мягкую проверку безопасности
                if aggressive:
                    # Проверяем, можем ли мы хотя бы отойти на одну клетку
                    can_escape = False
                    escape_path = None
                    for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                        if self.is_valid_position(escape_pos):
                            # Проверяем, что escape позиция не в радиусе взрыва
                            if escape_pos.manhattan_distance(bomb_pos) >= self.bomb_range + 1:
                                # Проверяем, что escape позиция безопасна от других бомб
                                if self.get_blast_danger(escape_pos) > 1.0:
                                    escape_path = [bomber.pos, escape_pos]
                                    can_escape = True
                                    break
                    
                    if can_escape and escape_path:
                        return (bomb_pos, escape_path)
                else:
                    # Обычная проверка безопасности (но все равно проверяем, заперт ли юнит)
                    is_enclosed = self.is_bomber_enclosed(bomber)
                    can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=is_enclosed)
                    if can_place and escape_path:
                        return (bomb_pos, escape_path)
        
        return None
        
    def get_best_target(self, bomber: Bomber, assigned_targets: Set[Position] = None, bomber_index: int = 0) -> Optional[Position]:
        """
        Улучшенный выбор цели с учетом специализации бомберов.
        Использует TargetSelector для поиска лучших позиций бомб.
        """
        if assigned_targets is None:
            assigned_targets = set()

        # Используем продвинутую логику, если доступна
        if self.use_advanced_logic and self.target_selector:
            # Получаем видимые препятствия (в радиусе видимости)
            visible_obstacles = set()
            vision_range = 5 + (3 * (self.booster_manager.speed_level if self.booster_manager else 0))
            
            for obs in self.obstacles:
                dist = bomber.pos.manhattan_distance(obs)
                if dist <= vision_range:
                    visible_obstacles.add((obs.x, obs.y))
            
            # Находим лучшие позиции для бомб
            best_spots = self.target_selector.find_best_bomb_spots(
                (bomber.pos.x, bomber.pos.y),
                visible_obstacles,
                max_distance=15
            )
            
            # Выбираем лучшую позицию, которая не назначена другим юнитам
            for (pos, score, box_count) in best_spots:
                target_pos = Position(pos[0], pos[1])
                if target_pos not in assigned_targets:
                    # Проверяем, что позиция доступна
                    if self.is_valid_position(target_pos):
                        print(f"[TARGET] Selected advanced target {target_pos} (score={score}, boxes={box_count})")
                        return target_pos

        possible_targets = []

        # 1. Анализ карты для кластеров (высший приоритет)
        if self.use_local_api and self.api:
            analysis = self.api.get_map_analysis(force_update=False)
            if analysis and analysis.high_value_targets:
                for pos_tuple, val in analysis.high_value_targets[:15]:
                    pos = Position(pos_tuple[0], pos_tuple[1])
                    dist = bomber.pos.distance(pos)
                    if dist < 30:  # Увеличиваем радиус поиска
                        # Специализация: первый бомбер берет самые крупные кластеры
                        cluster_bonus = val * 2 if bomber_index == 0 and val >= 4 else val
                        penalty = 100 if pos in assigned_targets else 0
                        possible_targets.append((pos, cluster_bonus * 15 - penalty))

        # 2. Враги и мобы (высокий приоритет)
        for enemy in self.enemies + [Mob(m.id, m.pos, m.safe_time, m.type) for m in self.mobs if not m.safe_time]:
            dist = bomber.pos.manhattan_distance(enemy.pos)
            if dist <= 12:  # Увеличиваем радиус обнаружения
                # Приоритет на призраков (ghost mobs)
                is_ghost = hasattr(enemy, 'type') and enemy.type == "ghost"
                base_priority = 50 if is_ghost else 25
                penalty = 60 if enemy.pos in assigned_targets else 0
                possible_targets.append((enemy.pos, base_priority - penalty))

        # 3. Одиночные препятствия (низкий приоритет, но с координацией)
        if self.obstacles:
            # Распределяем препятствия между бомберами по зонам
            bomber_zone_x = (bomber_index * self.map_size[0]) // len(self.bombers) if self.map_size else 0
            next_zone_x = ((bomber_index + 1) * self.map_size[0]) // len(self.bombers) if self.map_size else self.map_size[0] if self.map_size else 30

            zone_obstacles = [obs for obs in self.obstacles if bomber_zone_x <= obs.x < next_zone_x]
            sample_obstacles = random.sample(zone_obstacles, min(15, len(zone_obstacles))) if zone_obstacles else []

            for obs in sample_obstacles:
                # Ищем соседей для оценки ценности
                neighbors = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                              if Position(obs.x + dx, obs.y + dy) in self.obstacles)
                value = 3 + neighbors  # Чем больше соседей, тем ценнее
                penalty = 20 if obs in assigned_targets else 0
                possible_targets.append((obs, value - penalty))

        if not possible_targets:
            return None

        # Улучшенный выбор с учетом эффективности и риска
        best_target = None
        best_score = -float('inf')

        for target_pos, base_score in possible_targets:
            if target_pos in assigned_targets and base_score < 20:
                continue  # Пропускаем назначенные цели, кроме очень ценных

            dist = bomber.pos.manhattan_distance(target_pos)

            # Расчетное время достижения цели
            speed = 2 + (self.booster_manager.speed_level if self.booster_manager else 0)  # клетки/сек
            time_to_target = dist / speed

            # Оценка риска (опасность на пути)
            danger_level = self.get_blast_danger(target_pos, time_offset=time_to_target)

            # Штраф за опасность
            danger_penalty = 0
            if danger_level < time_to_target:
                danger_penalty = 50  # Высокий штраф за опасные цели
            elif danger_level < time_to_target + 1.0:
                danger_penalty = 20  # Средний штраф

            # Бонус за близость
            proximity_bonus = max(0, 15 - dist)

            # Финальный скор
            score = base_score + proximity_bonus - danger_penalty

            if score > best_score:
                best_score = score
                best_target = target_pos

        return best_target

    def find_chain_bomb_opportunity(self, bomber: Bomber, bomber_idx: int, assigned_targets: Set[Position]) -> Optional[Tuple[Position, List[Position]]]:
        """Найти возможность для цепного взрыва с другими бомберами"""
        # Ищем позицию, где бомба этого бомбера может создать цепную реакцию
        # с бомбами других бомберов или с препятствиями

        # Проверяем, есть ли уже установленные бомбы рядом
        for bomb in self.bombs:
            # Если бомба взорвется через 3-6 секунд, можем создать цепь
            if 3.0 <= bomb.timer <= 6.0:
                # Ищем позицию рядом с этой бомбой для нашего взрыва
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    chain_pos = Position(bomb.pos.x + dx, bomb.pos.y + dy)
                    if (self.is_valid_position(chain_pos) and
                        chain_pos not in assigned_targets and
                        bomber.pos.manhattan_distance(chain_pos) <= 8):

                        # Проверяем, можем ли безопасно поставить бомбу
                        can_place, escape_path = self.can_safely_place_bomb(bomber, chain_pos)
                        if can_place and escape_path:
                            # Проверяем, что наш взрыв попадет под радиус бомбы другого игрока
                            if (abs(chain_pos.x - bomb.pos.x) <= self.bomb_range and
                                chain_pos.y == bomb.pos.y) or \
                               (abs(chain_pos.y - bomb.pos.y) <= self.bomb_range and
                                chain_pos.x == bomb.pos.x):
                                return (chain_pos, escape_path)

        # Ищем возможность создать "бомбовый коридор" для других бомберов
        if bomber_idx == 0:  # Только первый бомбер создает коридоры
            for other_bomber in self.bombers[1:]:
                if not other_bomber.alive or other_bomber.bombs_available == 0:
                    continue

                # Ищем препятствия на пути другого бомбера
                path_to_other = self.find_path(bomber.pos, other_bomber.pos, max_steps=15)
                if path_to_other and len(path_to_other) > 3:
                    # Ищем препятствия на этом пути
                    for pos in path_to_other[1:-1]:  # Пропускаем старт и финиш
                        if pos in self.obstacles:
                            # Проверяем, можем ли мы создать проход
                            can_place, escape_path = self.can_safely_place_bomb(bomber, pos)
                            if can_place and escape_path and pos not in assigned_targets:
                                return (pos, escape_path)

        return None

    def make_move(self) -> Dict:
        """Основная логика хода согласно новому алгоритму"""
        if not self.strategy:
            return {}
        
        commands = []
        alive_bombers = [b for b in self.bombers if b.alive and b.can_move]
        
        if not alive_bombers:
            return {}
        
        # Инкрементируем счетчик спавна
        self.strategy.increment_spawn_iterations()
        
        # ФАЗА СПАВНА (первые 10 итераций)
        if self.strategy.is_spawn_phase():
            result = self.strategy.handle_spawn_phase(alive_bombers)
            if result:
                return result
        
        # Если остался один бомбер
        if len(alive_bombers) == 1:
            command = self.strategy.handle_single_bomber(alive_bombers[0])
            if command:
                commands.append(command)
            if commands:
                return self.move_bombers(commands)
            return {}
        
        # Организуем группы по 2 бомбера
        self.strategy.organize_groups(alive_bombers)
        
        # Собираем назначенные цели заранее (для всех бомберов)
        # Используем кортежи (x, y) для сериализации
        assigned_targets = set()
        assigned_enemies = set()  # Назначенные враги (id)
        assigned_mobs = set()  # Назначенные мобы (id)
        
        # Обрабатываем каждого бомбера
        for bomber in alive_bombers:
            my_command = {"id": bomber.id, "path": [], "bombs": []}
            command_added = False
            
            # ПРИОРИТЕТ 3.5: Скрипт избегания бомбы за 2 секунды до взрыва
            avoidance_path = self.strategy.check_bomb_avoidance(bomber)
            if avoidance_path:
                path_list = [[p.x, p.y] for p in avoidance_path[:5]]
                validated_path = self._validate_path(path_list, bomber.pos)
                if validated_path:
                    my_command["path"] = validated_path
                    commands.append(my_command)
                    print(f"[BOMB AVOIDANCE] {bomber.id} avoiding bomb")
                    command_added = True
                    continue
            
            # Спасаем других бомберов за 2 секунды до взрыва
            for other_bomber in self.bombers:
                if other_bomber.id == bomber.id or not other_bomber.alive:
                    continue
                other_danger = self.get_blast_danger(other_bomber.pos)
                if other_danger <= self.strategy.bomb_avoidance_time:
                    # Помогаем другому бомберу убежать
                    help_path = self.strategy.check_bomb_avoidance(other_bomber)
                    if help_path:
                        path_list = [[p.x, p.y] for p in help_path[:3]]
                        validated_path = self._validate_path(path_list, bomber.pos)
                        if validated_path:
                            my_command["path"] = validated_path
                            commands.append(my_command)
                            print(f"[HELP] {bomber.id} helping {other_bomber.id} escape bomb")
                            command_added = True
                            break
            
            if command_added:
                continue
            
            # ПРИОРИТЕТ 2: Мобы
            nearby_ghost = self.find_nearby_ghost(bomber, max_distance=10)
            if nearby_ghost:
                mob_command = self.strategy.handle_mob(bomber, nearby_ghost)
                if mob_command:
                    # Валидируем путь в команде
                    if mob_command.get("path"):
                        path_list = mob_command["path"]
                        validated_path = self._validate_path(path_list, bomber.pos)
                        if validated_path:
                            mob_command["path"] = validated_path
                            commands.append(mob_command)
                            print(f"[MOB] {bomber.id} handling ghost at {nearby_ghost.pos}")
                            continue
            
            # Проверяем патрулей
            for mob in self.mobs:
                if mob.safe_time > 0 or mob.type != "patrol" or mob.id in assigned_mobs:
                    continue
                dist = bomber.pos.manhattan_distance(mob.pos)
                if dist <= 12:  # Увеличен радиус
                    mob_command = self.strategy.handle_mob(bomber, mob)
                    if mob_command:
                        # Валидируем путь в команде
                        if mob_command.get("path"):
                            path_list = mob_command["path"]
                            validated_path = self._validate_path(path_list, bomber.pos)
                            if validated_path:
                                mob_command["path"] = validated_path
                                commands.append(mob_command)
                                assigned_mobs.add(mob.id)  # Отмечаем моба как назначенного
                                print(f"[MOB] {bomber.id} handling patrol at {mob.pos}")
                                command_added = True
                                break
            
            if command_added:
                continue
            
            # ПРИОРИТЕТ 3: Вражеские бомберы (не свои!)
            nearby_enemy = self.find_nearby_enemy(bomber, max_distance=8)
            if nearby_enemy:
                enemy_dist = bomber.pos.manhattan_distance(nearby_enemy.pos)
                print(f"[ENEMY_CHECK] {bomber.id} found enemy at {nearby_enemy.pos}, distance={enemy_dist}, bombs_available={bomber.bombs_available}")
                enemy_command = self.strategy.handle_enemy_bomber(bomber, nearby_enemy)
                if enemy_command:
                    # Валидируем путь в команде
                    if enemy_command.get("path"):
                        path_list = enemy_command["path"]
                        validated_path = self._validate_path(path_list, bomber.pos)
                        if validated_path:
                            enemy_command["path"] = validated_path
                            commands.append(enemy_command)
                            assigned_enemies.add(nearby_enemy.id)  # Отмечаем врага как назначенного
                            print(f"[ENEMY] {bomber.id} handling enemy bomber at {nearby_enemy.pos}, placing bomb at {enemy_command.get('bombs', [])}")
                            continue
                        else:
                            print(f"[ENEMY] {bomber.id} enemy command path validation failed: {path_list}")
                            # Если путь невалиден, но есть бомба - ставим на месте
                            if enemy_command.get("bombs"):
                                fallback_command = {
                                    "id": bomber.id,
                                    "bombs": enemy_command["bombs"],
                                    "path": [[bomber.pos.x, bomber.pos.y]]
                                }
                                commands.append(fallback_command)
                                print(f"[ENEMY] {bomber.id} placing bomb at {enemy_command.get('bombs')} without movement")
                                continue
                    else:
                        print(f"[ENEMY] {bomber.id} enemy command has no path")
                        # Если есть бомба, ставим на месте
                        if enemy_command.get("bombs"):
                            fallback_command = {
                                "id": bomber.id,
                                "bombs": enemy_command["bombs"],
                                "path": [[bomber.pos.x, bomber.pos.y]]
                            }
                            commands.append(fallback_command)
                            print(f"[ENEMY] {bomber.id} placing bomb at {enemy_command.get('bombs')} without path")
                            continue
                else:
                    print(f"[ENEMY] {bomber.id} handle_enemy_bomber returned None - enemy too far, moving closer")
                    # Враг слишком далеко - делаем шаг навстречу (только горизонтальные/вертикальные!)
                    enemy_dist = bomber.pos.manhattan_distance(nearby_enemy.pos)
                    if enemy_dist > self.bomb_range:
                        # Двигаемся к врагу (только один шаг за раз!)
                        dx = nearby_enemy.pos.x - bomber.pos.x
                        dy = nearby_enemy.pos.y - bomber.pos.y
                        
                        # Выбираем направление (приоритет X, потом Y)
                        move_dx, move_dy = 0, 0
                        if dx != 0:
                            move_dx = 1 if dx > 0 else -1
                        elif dy != 0:
                            move_dy = 1 if dy > 0 else -1
                        
                        move_pos = Position(bomber.pos.x + move_dx, bomber.pos.y + move_dy)
                        if self.is_valid_position(move_pos):
                            move_command = {
                                "id": bomber.id,
                                "path": [[move_pos.x, move_pos.y]],
                                "bombs": []
                            }
                            commands.append(move_command)
                            print(f"[ENEMY] {bomber.id} moving closer to enemy at {nearby_enemy.pos}")
                            continue
            
            # ПРИОРИТЕТ 5: Агрессивное уничтожение препятствий
            # Сначала проверяем, можем ли мы поставить бомбу прямо сейчас для уничтожения препятствий
            if bomber.bombs_available > 0:
                # Ищем ближайшие препятствия в радиусе бомбы
                nearby_obstacles = []
                for obs in self.obstacles:
                    dist = bomber.pos.manhattan_distance(obs)
                    if dist <= self.bomb_range + 1:  # В радиусе бомбы + 1 клетка
                        # Проверяем, может ли бомба на текущей позиции достать до препятствия
                        can_reach = False
                        for r_dx, r_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            for r in range(1, self.bomb_range + 1):
                                check_pos = Position(bomber.pos.x + r_dx * r, bomber.pos.y + r_dy * r)
                                if check_pos == obs:
                                    can_reach = True
                                    break
                                if check_pos in self.walls:
                                    break
                                if check_pos in self.obstacles and check_pos != obs:
                                    break
                            if can_reach:
                                break
                        
                        if can_reach:
                            nearby_obstacles.append((obs, dist))
                
                # Сортируем по расстоянию (ближайшие первыми)
                nearby_obstacles.sort(key=lambda x: x[1])
                
                # Пробуем поставить бомбу для уничтожения ближайшего препятствия
                for obs, dist in nearby_obstacles[:3]:  # Проверяем до 3 ближайших
                    bomb_pos = bomber.pos
                    
                    # Проверяем, не попадет ли бомба в других бомберов
                    will_hit_others = self.will_bomb_hit_other_bombers(bomb_pos, bomber.id)
                    if not will_hit_others:
                        # Ставим бомбу на текущей позиции
                        my_command["bombs"] = [[bomb_pos.x, bomb_pos.y]]
                        # Ищем путь для отхода (агрессивно)
                        can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=True)
                        
                        if escape_path:
                            escape_path_list = [[p.x, p.y] for p in escape_path[:8]]
                            validated_escape = self._validate_path(escape_path_list, bomber.pos)
                            if validated_escape:
                                my_command["path"] = validated_escape
                            else:
                                my_command["path"] = [[bomber.pos.x, bomber.pos.y]]
                        else:
                            # Нет escape_path - все равно ставим бомбу (агрессивно)
                            my_command["path"] = [[bomber.pos.x, bomber.pos.y]]
                        
                        assigned_targets.add((obs.x, obs.y))
                        commands.append(my_command)
                        print(f"[BOMB] {bomber.id} placing bomb at current pos {bomb_pos} to destroy obstacle at {obs} (distance={dist})")
                        continue
            
            # ПРИОРИТЕТ 6: Движение к стенам для дальнейшего уничтожения
            target = self.strategy.find_best_wall_target(bomber, assigned_targets)
            if target:
                
                # Если не поставили бомбу, двигаемся к цели
                path = self.find_path(bomber.pos, target, max_steps=15)
                if path and len(path) > 1:
                    path_list = [[p.x, p.y] for p in path[:15]]
                    validated_path = self._validate_path(path_list, bomber.pos)
                    if not validated_path:
                        # Путь невалиден, пробуем простой путь
                        simple_path = self._create_simple_path(bomber.pos, target)
                        if simple_path and len(simple_path) > 1:
                            path_list = [[p.x, p.y] for p in simple_path[:10]]
                            validated_path = self._validate_path(path_list, bomber.pos)
                    
                    if validated_path:
                        # Проверяем безопасность пути - избегаем позиций с бомбами
                        safe_path = []
                        for step_pos in validated_path:
                            step_pos_obj = Position(step_pos[0], step_pos[1])
                            # Проверяем опасность с учетом времени прибытия
                            step_index = validated_path.index(step_pos)
                            arrival_time = step_index * 0.5  # Время до прибытия на эту позицию
                            danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time)
                            
                            # Если позиция опасна (бомба взорвется пока мы там), останавливаемся перед ней
                            if danger < 4.0:  # Опасность через менее 4 секунд
                                break
                            
                            safe_path.append(step_pos)
                        
                        # Если путь стал короче из-за опасности, используем безопасную часть
                        if safe_path:
                            validated_path = safe_path
                        else:
                            # Если весь путь опасен, ищем альтернативный путь
                            continue
                        
                        my_command["path"] = validated_path
                        
                        # АКТИВНО: Ставим бомбы при движении, если есть препятствия рядом
                        if bomber.bombs_available > 0:
                            bomb_placed = False
                            
                            # Проверяем препятствия рядом с путем
                            for step_pos in validated_path[:3]:  # Проверяем первые 3 шага пути
                                step_pos_obj = Position(step_pos[0], step_pos[1])
                                
                                # Проверяем, есть ли препятствия в радиусе бомбы
                                for obs in self.obstacles:
                                    dist = step_pos_obj.manhattan_distance(obs)
                                    if dist <= self.bomb_range + 1:
                                        # Проверяем, может ли бомба достать до препятствия
                                        can_reach = False
                                        for r_dx, r_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                            for r in range(1, self.bomb_range + 1):
                                                check_pos = Position(step_pos_obj.x + r_dx * r, step_pos_obj.y + r_dy * r)
                                                if check_pos == obs:
                                                    can_reach = True
                                                    break
                                                if check_pos in self.walls:
                                                    break
                                                if check_pos in self.obstacles and check_pos != obs:
                                                    break
                                            if can_reach:
                                                break
                                        
                                        if can_reach:
                                            # Проверяем безопасность установки бомбы
                                            will_hit_others = self.will_bomb_hit_other_bombers(step_pos_obj, bomber.id)
                                            if not will_hit_others:
                                                # Проверяем, можем ли убежать от бомбы
                                                step_index = validated_path.index(step_pos)
                                                arrival_time = step_index * 0.5
                                                future_danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time + 4.0)
                                                
                                                # Если можем убежать (опасность через >4 секунд после установки), ставим бомбу
                                                if future_danger > 4.0 or len(validated_path) > step_index + 2:
                                                    my_command["bombs"] = [step_pos]
                                                    print(f"[BOMB] {bomber.id} placing bomb at {step_pos} while moving (obstacle at {obs})")
                                                    bomb_placed = True
                                                    break
                                
                                if bomb_placed:
                                    break
                            
                            # АКТИВНО: Ставим бомбы при движении (бомбовый след) - даже если нет препятствий рядом
                            if not bomb_placed and len(validated_path) > 1:
                                current_danger = self.get_blast_danger(bomber.pos)
                                if current_danger > 5.0:  # Текущая позиция безопасна
                                    will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                                    if not will_hit_others:
                                        # Проверяем, есть ли препятствия рядом с текущей позицией
                                        nearby_obstacles = [obs for obs in self.obstacles 
                                                          if bomber.pos.manhattan_distance(obs) <= self.bomb_range + 3]
                                        
                                        # Ставим бомбу если есть препятствия ИЛИ если путь длинный (бомбовый след)
                                        if nearby_obstacles or len(validated_path) >= 3:
                                            # Проверяем безопасность следующей позиции
                                            next_pos = Position(validated_path[1][0], validated_path[1][1])
                                            next_danger = self.get_blast_danger(next_pos, time_offset=0.5)
                                            if next_danger > 4.0:  # Следующая позиция безопасна
                                                my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                print(f"[BOMB] {bomber.id} placing bomb trail at {bomber.pos} while moving")
                        
                        # Добавляем в назначенные (используем кортеж для сериализации)
                        assigned_targets.add((target.x, target.y))
                        commands.append(my_command)
                        print(f"[MAIN] {bomber.id} moving to target at {target}")
                        continue
                elif path and len(path) == 1:
                    # Уже на месте цели - АКТИВНО двигаемся дальше
                    # Ищем следующую цель или двигаемся в случайном направлении
                    next_target = self.strategy.find_best_wall_target(bomber, assigned_targets)
                    if next_target and next_target != bomber.pos:
                        # Пробуем найти путь к другой цели
                        path = self.find_path(bomber.pos, next_target, max_steps=15)
                        if path and len(path) > 1:
                            path_list = [[p.x, p.y] for p in path[:15]]
                            validated_path = self._validate_path(path_list, bomber.pos)
                            if validated_path:
                                my_command["path"] = validated_path
                                
                                # Ставим бомбу при движении
                                if bomber.bombs_available > 0:
                                    current_danger = self.get_blast_danger(bomber.pos)
                                    if current_danger > 5.0:
                                        will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                                        if not will_hit_others:
                                            next_pos = Position(validated_path[1][0], validated_path[1][1])
                                            next_danger = self.get_blast_danger(next_pos, time_offset=0.5)
                                            if next_danger > 4.0:
                                                my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                print(f"[BOMB] {bomber.id} placing bomb trail when moving to next target")
                                
                                assigned_targets.add((next_target.x, next_target.y))
                                commands.append(my_command)
                                print(f"[MAIN] {bomber.id} already at target, moving to {next_target}")
                                continue
                    else:
                        # Нет другой цели - двигаемся в случайном безопасном направлении
                        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        random.shuffle(directions)
                        for dx, dy in directions:
                            move_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                            if self.is_valid_position(move_pos):
                                danger = self.get_blast_danger(move_pos, time_offset=0.5)
                                if danger > 4.0:
                                    my_command["path"] = [[bomber.pos.x, bomber.pos.y], [move_pos.x, move_pos.y]]
                                    
                                    # Ставим бомбу при движении
                                    if bomber.bombs_available > 0:
                                        current_danger = self.get_blast_danger(bomber.pos)
                                        if current_danger > 5.0:
                                            will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                                            if not will_hit_others:
                                                my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                print(f"[BOMB] {bomber.id} placing bomb trail when moving randomly")
                                    
                                    commands.append(my_command)
                                    print(f"[ACTIVE] {bomber.id} moving randomly to {move_pos}")
                                    continue
                else:
                    # Путь не найден - пробуем более простой путь (прямой)
                    if bomber.pos.manhattan_distance(target) <= 20:
                        # Пробуем простой путь по прямой
                        simple_path = self._create_simple_path(bomber.pos, target)
                        if simple_path and len(simple_path) > 1:
                            # Конвертируем в список списков
                            path_list = [[p.x, p.y] for p in simple_path[:10]]
                            
                            # Проверяем безопасность пути
                            safe_path = []
                            for step_pos in path_list:
                                step_pos_obj = Position(step_pos[0], step_pos[1])
                                step_index = path_list.index(step_pos)
                                arrival_time = step_index * 0.5
                                danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time)
                                
                                if danger < 4.0:
                                    break
                                
                                safe_path.append(step_pos)
                            
                            if safe_path:
                                validated_path = self._validate_path(safe_path, bomber.pos)
                            else:
                                validated_path = self._validate_path(path_list, bomber.pos)
                            
                            if validated_path:
                                my_command["path"] = validated_path
                                
                                # АКТИВНО: Ставим бомбы при движении (бомбовый след)
                                if bomber.bombs_available > 0:
                                    bomb_placed = False
                                    
                                    # Проверяем препятствия рядом с путем
                                    for step_pos in validated_path[:3]:
                                        step_pos_obj = Position(step_pos[0], step_pos[1])
                                        nearby_obstacles = [obs for obs in self.obstacles 
                                                          if step_pos_obj.manhattan_distance(obs) <= self.bomb_range + 1]
                                        if nearby_obstacles:
                                            will_hit_others = self.will_bomb_hit_other_bombers(step_pos_obj, bomber.id)
                                            if not will_hit_others:
                                                step_index = validated_path.index(step_pos)
                                                arrival_time = step_index * 0.5
                                                future_danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time + 4.0)
                                                if future_danger > 4.0:
                                                    my_command["bombs"] = [step_pos]
                                                    print(f"[BOMB] {bomber.id} placing bomb at {step_pos} on simple path")
                                                    bomb_placed = True
                                                    break
                                    
                                    # Если не поставили для препятствия, ставим на текущей позиции (бомбовый след)
                                    if not bomb_placed and len(validated_path) > 1:
                                        current_danger = self.get_blast_danger(bomber.pos)
                                        if current_danger > 5.0:
                                            will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                                            if not will_hit_others:
                                                next_pos = Position(validated_path[1][0], validated_path[1][1])
                                                next_danger = self.get_blast_danger(next_pos, time_offset=0.5)
                                                if next_danger > 4.0:
                                                    my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                                    print(f"[BOMB] {bomber.id} placing bomb trail at {bomber.pos} on simple path")
                                
                                assigned_targets.add((target.x, target.y))
                                commands.append(my_command)
                                print(f"[MAIN] {bomber.id} using simple path to {target}")
                                continue
            
            # Fallback: ищем любое движение (даже если нет препятствий)
            fallback_target = self._find_fallback_movement(bomber, assigned_targets)
            if fallback_target:
                path = self.find_path(bomber.pos, fallback_target, max_steps=10)
                if path and len(path) > 1:
                    path_list = [[p.x, p.y] for p in path[:10]]
                    
                    # Проверяем безопасность пути
                    safe_path = []
                    for step_pos in path_list:
                        step_pos_obj = Position(step_pos[0], step_pos[1])
                        step_index = path_list.index(step_pos)
                        arrival_time = step_index * 0.5
                        danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time)
                        
                        # Если позиция опасна, останавливаемся перед ней
                        if danger < 4.0:
                            break
                        
                        safe_path.append(step_pos)
                    
                    if safe_path:
                        validated_path = self._validate_path(safe_path, bomber.pos)
                    else:
                        validated_path = self._validate_path(path_list, bomber.pos)
                    
                    if validated_path:
                        my_command["path"] = validated_path
                        
                        # АКТИВНО: Ставим бомбы при движении (бомбовый след)
                        if bomber.bombs_available > 0:
                            bomb_placed = False
                            
                            # Проверяем препятствия рядом с путем
                            for step_pos in validated_path[:3]:
                                step_pos_obj = Position(step_pos[0], step_pos[1])
                                nearby_obstacles = [obs for obs in self.obstacles 
                                                  if step_pos_obj.manhattan_distance(obs) <= self.bomb_range + 1]
                                if nearby_obstacles:
                                    will_hit_others = self.will_bomb_hit_other_bombers(step_pos_obj, bomber.id)
                                    if not will_hit_others:
                                        step_index = validated_path.index(step_pos)
                                        arrival_time = step_index * 0.5
                                        future_danger = self.get_blast_danger(step_pos_obj, time_offset=arrival_time + 4.0)
                                        if future_danger > 4.0:
                                            my_command["bombs"] = [step_pos]
                                            print(f"[BOMB] {bomber.id} placing bomb at {step_pos} during fallback movement")
                                            bomb_placed = True
                                            break
                            
                            # Если не поставили для препятствия, ставим на текущей позиции (бомбовый след)
                            if not bomb_placed and len(validated_path) > 1:
                                current_danger = self.get_blast_danger(bomber.pos)
                                if current_danger > 5.0:
                                    will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                                    if not will_hit_others:
                                        next_pos = Position(validated_path[1][0], validated_path[1][1])
                                        next_danger = self.get_blast_danger(next_pos, time_offset=0.5)
                                        if next_danger > 4.0:
                                            my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                            print(f"[BOMB] {bomber.id} placing bomb trail at {bomber.pos} during fallback")
                        
                        commands.append(my_command)
                        print(f"[FALLBACK] {bomber.id} moving to {fallback_target}")
                        continue
            
            # Последний fallback: АКТИВНОЕ движение - всегда двигаемся, даже если нет целей
            # Ищем безопасное направление для движения
            best_move = None
            best_safety = -1
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)  # Случайный порядок для разнообразия
            
            for dx, dy in directions:
                move_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                if not self.is_valid_position(move_pos):
                    continue
                
                # Проверяем безопасность позиции
                danger = self.get_blast_danger(move_pos, time_offset=0.5)
                
                # Предпочитаем позиции дальше от других бомберов
                min_dist_to_others = float('inf')
                for other in alive_bombers:
                    if other.id == bomber.id:
                        continue
                    dist = other.pos.manhattan_distance(move_pos)
                    min_dist_to_others = min(min_dist_to_others, dist)
                
                # Комплексный счет: безопасность + расстояние от других
                safety_score = danger + min_dist_to_others * 0.1
                
                if safety_score > best_safety:
                    best_safety = safety_score
                    best_move = move_pos
            
            if best_move:
                my_command["path"] = [[bomber.pos.x, bomber.pos.y], [best_move.x, best_move.y]]
                
                # АКТИВНО: Ставим бомбу при движении (бомбовый след)
                if bomber.bombs_available > 0:
                    current_danger = self.get_blast_danger(bomber.pos)
                    if current_danger > 5.0:  # Текущая позиция безопасна
                        will_hit_others = self.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                        if not will_hit_others:
                            # Проверяем, можем ли убежать
                            move_danger = self.get_blast_danger(best_move, time_offset=0.5)
                            if move_danger > 4.0:  # Позиция движения безопасна
                                my_command["bombs"] = [[bomber.pos.x, bomber.pos.y]]
                                print(f"[BOMB] {bomber.id} placing bomb trail at {bomber.pos} while moving to {best_move}")
                
                commands.append(my_command)
                print(f"[ACTIVE] {bomber.id} actively moving to {best_move}")
            else:
                # Если не можем двигаться, хотя бы стоим (но это крайний случай)
                my_command["path"] = [[bomber.pos.x, bomber.pos.y]]
                commands.append(my_command)
        
        if commands:
            # Убеждаемся, что все пути валидны и нет объектов Position
            validated_commands = []
            for cmd in commands:
                # Находим позицию бомбера для валидации
                bomber_id = cmd.get("id")
                bomber = next((b for b in self.bombers if b.id == bomber_id), None)
                if not bomber:
                    continue
                
                validated_cmd = {
                    "id": str(bomber_id),  # Убеждаемся, что это строка
                    "path": [],
                    "bombs": []
                }
                
                # Обрабатываем path
                if cmd.get("path"):
                    path_raw = cmd["path"]
                    # Конвертируем все элементы в списки [x, y]
                    path_list = []
                    for p in path_raw:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            path_list.append([int(p[0]), int(p[1])])
                        elif hasattr(p, 'x') and hasattr(p, 'y'):  # Position объект
                            path_list.append([int(p.x), int(p.y)])
                    
                    # Валидируем путь
                    if path_list:
                        validated_path = self._validate_path(path_list, bomber.pos)
                        if validated_path:
                            validated_cmd["path"] = validated_path
                        else:
                            # Если путь невалиден, оставляем только текущую позицию
                            validated_cmd["path"] = [[bomber.pos.x, bomber.pos.y]]
                    else:
                        validated_cmd["path"] = [[bomber.pos.x, bomber.pos.y]]
                else:
                    validated_cmd["path"] = [[bomber.pos.x, bomber.pos.y]]
                
                # Обрабатываем bombs
                if cmd.get("bombs"):
                    bombs_raw = cmd["bombs"]
                    bombs_list = []
                    for b in bombs_raw:
                        if isinstance(b, (list, tuple)) and len(b) >= 2:
                            bombs_list.append([int(b[0]), int(b[1])])
                        elif hasattr(b, 'x') and hasattr(b, 'y'):  # Position объект
                            bombs_list.append([int(b.x), int(b.y)])
                    validated_cmd["bombs"] = bombs_list
                
                validated_commands.append(validated_cmd)
            
            if validated_commands:
                print(f"[MOVE] Sending {len(validated_commands)} commands")
                for cmd in validated_commands:
                    if cmd.get("bombs"):
                        print(f"  [BOMB] {cmd['id']} placing bomb at {cmd['bombs'][0]}")
                return self.move_bombers(validated_commands)
        
        return {}
    
    def _find_fallback_movement(self, bomber: Bomber, assigned_targets: Set) -> Optional[Position]:
        """Найти fallback цель для АКТИВНОГО движения - всегда находим куда двигаться"""
        import random
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        
        best_pos = None
        best_score = -1
        
        # Ищем лучшую позицию с учетом безопасности и расстояния от других бомберов
        for radius in range(1, 8):  # Ищем в радиусе до 8 клеток
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    
                    check_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if not self.is_valid_position(check_pos):
                        continue
                    if check_pos in self.walls:
                        continue
                    if (check_pos.x, check_pos.y) in assigned_targets:
                        continue
                    
                    # Проверяем безопасность позиции
                    danger = self.get_blast_danger(check_pos, time_offset=radius * 0.5)
                    
                    # Предпочитаем позиции дальше от других бомберов
                    min_dist_to_others = float('inf')
                    for other in self.bombers:
                        if other.id == bomber.id or not other.alive:
                            continue
                        dist = other.pos.manhattan_distance(check_pos)
                        min_dist_to_others = min(min_dist_to_others, dist)
                    
                    # Проверяем, есть ли препятствия рядом (для установки бомб)
                    nearby_obstacles = [obs for obs in self.obstacles 
                                      if check_pos.manhattan_distance(obs) <= self.bomb_range + 2]
                    
                    # Комплексный счет: безопасность + расстояние от других + наличие препятствий
                    score = danger + min_dist_to_others * 0.2 + len(nearby_obstacles) * 5
                    
                    # Небольшой штраф за слишком большое расстояние
                    if radius > 5:
                        score -= 10
                    
                    if score > best_score:
                        best_score = score
                        best_pos = check_pos
        
        return best_pos
    
    def _create_simple_path(self, start: Position, target: Position) -> Optional[List[Position]]:
        """Создать простой путь по прямой (для случаев, когда A* не находит путь)"""
        # Убеждаемся, что путь начинается с текущей позиции
        path = [start]
        current = start
        
        # Ограничиваем максимальное расстояние для простого пути
        max_distance = 15
        if start.manhattan_distance(target) > max_distance:
            return None
        
        steps = 0
        while current != target and steps < max_distance:
            # ВАЖНО: Нельзя двигаться по диагонали! Только по одной оси за раз
            dx = 0
            dy = 0
            
            # Приоритет: сначала по X, потом по Y
            if current.x != target.x:
                dx = 1 if current.x < target.x else -1
            elif current.y != target.y:
                dy = 1 if current.y < target.y else -1
            else:
                break  # Достигли цели
            
            # Двигаемся только по одной оси (не по диагонали!)
            if dx != 0:
                next_pos = Position(current.x + dx, current.y)
            elif dy != 0:
                next_pos = Position(current.x, current.y + dy)
            else:
                break
            
            # Проверяем валидность
            if not self.is_valid_position(next_pos) or next_pos in self.walls or next_pos in self.obstacles:
                # Если не можем двигаться в этом направлении, пробуем другое
                if dx != 0:
                    # Пробуем по Y
                    if current.y != target.y:
                        dy = 1 if current.y < target.y else -1
                        next_pos = Position(current.x, current.y + dy)
                        if not self.is_valid_position(next_pos) or next_pos in self.walls or next_pos in self.obstacles:
                            break
                    else:
                        break
                elif dy != 0:
                    # Пробуем по X
                    if current.x != target.x:
                        dx = 1 if current.x < target.x else -1
                        next_pos = Position(current.x + dx, current.y)
                        if not self.is_valid_position(next_pos) or next_pos in self.walls or next_pos in self.obstacles:
                            break
                    else:
                        break
                else:
                    break
            
            path.append(next_pos)
            current = next_pos
            steps += 1
        
        return path if len(path) > 1 else None
    
    def update_booster_stats(self):
        """Обновить статистику улучшений из состояния игры"""
        if self.booster_manager:
            self.booster_manager.update_stats()
            # Синхронизируем значения обратно в GameClient
            self.bomb_range = self.booster_manager.bomb_range
            self.bomb_delay = self.booster_manager.bomb_delay
    
    def should_buy_booster(self) -> Optional[int]:
        """Логика покупки бустеров"""
        if self.booster_manager:
            return self.booster_manager.should_buy_booster()
        return None

    def run(self, verbose: bool = False):
        """Главный цикл"""
        print("[CLIENT] Starting client run()...")
        iteration = 0
        last_stats_update = 0
        
        while True:
            try:
                iteration += 1
                current_time = time.time()

                # Обновляем состояние игры раз в 1 секунду (вместо каждые 0.5 сек)
                if current_time - last_stats_update > 1.0:
                    self.update_state()
                    last_stats_update = current_time

                # Обновляем статистику бустеров раз в 3 секунды
                if current_time - last_stats_update > 3.0:
                    self.update_booster_stats()

                # Покупка бустеров отключена
                result = self.make_move()
                if verbose and iteration % 10 == 0:
                    alive = len([b for b in self.bombers if b.alive])
                    obstacles = len(self.obstacles)
                    enemies = len(self.enemies)
                    mobs = len(self.mobs)
                    active_mobs = [m for m in self.mobs if m.safe_time == 0]
                    print(f"Iter {iteration}: Живых={alive}, Препятствий={obstacles}, Врагов={enemies}, Мобов={len(active_mobs)}/{mobs}")
                    
                    # Показываем позиции юнитов и ближайших врагов
                    for bomber in self.bombers[:2]:  # Первые 2 юнита
                        if bomber.alive:
                            nearby = self.find_nearby_enemy(bomber, max_distance=5)
                            if nearby:
                                dist = bomber.pos.manhattan_distance(nearby.pos)
                                print(f"  Юнит {bomber.id} на {bomber.pos}, враг на {nearby.pos} (расстояние {dist})")
                    
                    if result.get('errors'): print("Errors:", result['errors'])

                time.sleep(0.5) # Не частить
                
            except KeyboardInterrupt:
                break
            except requests.exceptions.HTTPError as e:
                # Не выводим ошибки 400 для boosters (они не критичны)
                if e.response and e.response.status_code == 400:
                    error_url = str(e.response.url) if hasattr(e.response, 'url') else str(e)
                    if "booster" in error_url.lower():
                        # Тихая обработка - просто пропускаем (не выводим вообще)
                        pass
                    elif verbose:
                        print(f"HTTP 400 Error: {e}")
                elif verbose:
                    print(f"HTTP Error: {e}")
                time.sleep(1)
            except Exception as e:
                # Не выводим ошибки, связанные с boosters (они не критичны)
                error_str = str(e)
                if "booster" not in error_str.lower() and "400" not in error_str:
                    print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    print("[MAIN] Starting game_client.py")
    import sys
    import os
    from dotenv import load_dotenv

    print("[MAIN] Loading .env...")
    load_dotenv(".env")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://games.datsteam.dev")

    print(f"[MAIN] API_KEY from .env: {api_key[:10] if api_key else 'None'}")
    print(f"[MAIN] BASE_URL: {base_url}")

    use_local_api = "--local" in sys.argv
    print(f"[MAIN] use_local_api: {use_local_api}")

    if not api_key:
        print("[MAIN] No API_KEY found, exiting")
        sys.exit(1)

    print("[MAIN] Creating client...")
    try:
        client = GameClient(api_key, base_url, use_local_api=use_local_api)
        print("[MAIN] Client created successfully")
    except Exception as e:
        print(f"[MAIN] Error creating client: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("[MAIN] Starting run()...")
    try:
        verbose = "--verbose" in sys.argv
        print(f"[MAIN] Verbose mode: {verbose}")
        client.run(verbose=verbose)
    except Exception as e:
        print(f"[MAIN] Error in run(): {e}")
        import traceback
        traceback.print_exc()

    use_local_api = False
    if "--local" in sys.argv: use_local_api = True

    print(f"[MAIN] Final config: API_KEY={api_key[:10] if api_key else 'None'}, BASE_URL={base_url}, local={use_local_api}")

    if not api_key:
        print("API_KEY required - create config.env or .env file")
        sys.exit(1)

    print("[MAIN] Creating GameClient...")
    client = GameClient(api_key, base_url, use_local_api=use_local_api)
    print("[MAIN] Starting client.run()...")
    client.run(verbose=True)
