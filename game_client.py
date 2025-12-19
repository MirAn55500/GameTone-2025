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
from dataclasses import dataclass
from enum import IntEnum

# Продвинутая логика с симуляцией цепных реакций
from advanced_logic import MapPredictor, TargetSelector, SafePathfinder


class BoosterType(IntEnum):
    """Типы улучшений"""
    POCKETS = 0  # +1 бомба
    BOMB_RANGE = 1  # +1 радиус
    SPEED = 2  # +1 скорость (макс 3)
    VISION = 3  # +3 обзор
    UNITS = 4  # +1 юнит (2 поинта)
    ARMOR = 5  # +1 броня
    BOMB_DELAY = 6  # -2 сек до взрыва (макс 3)
    ACROBATICS = 7  # проход через препятствия (2 поинта)


@dataclass
class Position:
    """Позиция на карте"""
    x: int
    y: int
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def distance(self, other: 'Position') -> float:
        """Расстояние до другой позиции"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def manhattan_distance(self, other: 'Position') -> int:
        """Манхэттенское расстояние"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


@dataclass
class Bomber:
    """Юнит игрока"""
    id: str
    pos: Position
    alive: bool
    can_move: bool
    bombs_available: int
    armor: int
    safe_time: int  # время неуязвимости в мс


@dataclass
class Bomb:
    """Бомба на карте"""
    pos: Position
    timer: float  # секунды до взрыва
    range: int  # радиус взрыва


@dataclass
class Enemy:
    """Враг"""
    id: str
    pos: Position
    safe_time: int


@dataclass
class Mob:
    """Моб"""
    id: str
    pos: Position
    type: str  # "ghost" или "patrol"
    safe_time: int


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
        
        # Статистика улучшений
        self.speed_level = 0  # максимум 3
        self.bomb_delay_level = 0  # максимум 3
        self.acrobatics_level = 0  # максимум 3
        self.bomb_range = 1  # текущий радиус бомбы
        self.bomb_delay = 8000  # время до взрыва в мс
        
        # Rate limiting
        self.last_request_time = 0
        self.request_times = []
        self.last_booster_429_error = 0
        self.booster_check_skip_until = 0
        

        # Статистика для отслеживания изменений
        self.last_enemy_count = 0
        self.last_mob_count = 0
        self.last_obstacle_count = 0
        self.last_points = 0
        self.total_kills = 0
        self.total_obstacles_destroyed = 0
        
        # Продвинутая логика (инициализируется после первого update_state)
        self.map_predictor: Optional[MapPredictor] = None
        self.target_selector: Optional[TargetSelector] = None
        self.safe_pathfinder: Optional[SafePathfinder] = None
        self.use_advanced_logic = True  # Флаг для включения/выключения продвинутой логики
    
    def _rate_limit(self):
        """Ограничение скорости запросов (3 в секунду)"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        if len(self.request_times) >= 3:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.request_times.append(time.time())
    
    def get_arena(self, retry_count: int = 3) -> Dict:
        """Получить состояние арены"""
        if self.use_local_api and self.api:
            return self.api.get_arena(use_cache=True)

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                response = self.session.get(f"{self.base_url}/api/arena")
                if response.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(0.5)
    
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
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                result = response.json()
                # Обрабатываем ошибки размещения бомб
                if result.get("errors"):
                    self._handle_bomb_placement_errors(result["errors"], commands)
                return result
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(0.5)
    
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
            
            # Каждый шаг должен быть соседней клеткой (максимум 1 клетка по X или Y)
            dx = abs(current_pos.x - prev_pos.x)
            dy = abs(current_pos.y - prev_pos.y)
            
            if dx > 1 or dy > 1 or (dx == 1 and dy == 1):
                print(f"[PATH VALIDATION] Invalid step: from {prev_pos} to {current_pos} (dx={dx}, dy={dy})")
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
            elif neighbor in self.obstacles and self.acrobatics_level < 2:
                blocked += 1
        return blocked >= 3

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
                        
                        if bomb_pos == obs and self.acrobatics_level < 2:
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
                if move_pos not in self.obstacles or self.acrobatics_level >= 2:
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
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(0.5)
    
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
        
        if self.safe_pathfinder is None:
            self.safe_pathfinder = SafePathfinder(
                self.map_predictor, walls_set, obstacles_set,
                enemies_data, mobs_data, speed=2 + self.speed_level
            )
        else:
            self.safe_pathfinder.predictor = self.map_predictor
            self.safe_pathfinder.walls = walls_set
            self.safe_pathfinder.obstacles = obstacles_set
            self.safe_pathfinder.enemies = enemies_data
            self.safe_pathfinder.mobs = mobs_data
            self.safe_pathfinder.speed = 2 + self.speed_level
    
    def is_valid_position(self, pos: Position) -> bool:
        """Проверить, находится ли позиция в пределах карты и не является ли стеной"""
        if not self.map_size: return False
        if not (0 <= pos.x < self.map_size[0] and 0 <= pos.y < self.map_size[1]):
            return False
        if pos in self.walls:
            return False
        # Препятствия блокируют движение, если нет акробатики
        if self.acrobatics_level < 2 and pos in self.obstacles:
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
        if start == target: return [target]
        
        # Используем продвинутую логику, если доступна
        if self.use_advanced_logic and self.safe_pathfinder:
            path = self.safe_pathfinder.find_path(
                (start.x, start.y), (target.x, target.y), max_ticks=max_steps * 2
            )
            if path:
                return [Position(x, y) for x, y in path]
            return None
        
        # Heuristic
        def h(p): return abs(p.x - target.x) + abs(p.y - target.y)
        
        open_set = [(h(start), 0, start, [start])] # f_score, g_score, current, path
        visited = {start: 0} # pos -> g_score
        
        start_time = time.time()
        
        while open_set:
            if time.time() - start_time > 0.1: # Timeout
                break
                
            f, g, current, path = heapq.heappop(open_set)
            
            if len(path) > max_steps: continue
            
            if current == target:
                return path
            
            # Neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                
                if not self.is_valid_position(neighbor): continue
                
                # Check danger from ALL bombs
                # Предполагаем скорость 2-3 клетки/сек. Время прибытия ~ g * 0.3
                arrival_time = (g + 1) * 0.3 
                danger_timer = self.get_blast_danger(neighbor, time_offset=0)
                
                cost = 1
                
                if danger_timer < float('inf'):
                    # КРИТИЧНО: избегаем позиций, где бомба взорвется пока мы там
                    if danger_timer < arrival_time + 2.0:  # Увеличили запас безопасности
                         if danger_timer > arrival_time - 0.5:  # Бомба взорвется пока мы там или сразу после
                             cost += 10000 # ОЧЕНЬ ОПАСНО, строго избегаем
                         elif danger_timer < arrival_time - 0.5:
                             # Взрывается до нас, но может быть опасно
                             cost += 100  # Средний штраф
                
                # КРИТИЧНО: Проверяем опасность от ВСЕХ бомб на карте
                # Если позиция в радиусе взрыва - строго избегаем
                neighbor_danger = self.get_blast_danger(neighbor, time_offset=arrival_time)
                if neighbor_danger < float('inf'):
                    # Позиция в радиусе взрыва
                    if neighbor_danger < arrival_time + 1.0:  # Бомба взорвется пока мы там
                        cost += 10000  # ОЧЕНЬ ОПАСНО - строго избегаем
                    elif neighbor_danger < arrival_time + 2.0:  # Бомба взорвется вскоре после прибытия
                        cost += 1000  # Опасно
                
                # Избегаем мобов (особенно призраков!)
                for mob in self.mobs:
                    if mob.safe_time > 0: continue  # Игнорируем спящих
                    
                    dist_to_mob = mob.pos.manhattan_distance(neighbor)
                    
                    if mob.pos == neighbor:
                        cost += 1000  # Контакт с мобом = смерть
                    elif mob.type == "ghost":
                        # Призраки опаснее - они могут проходить через препятствия
                        # Имеют радиус обзора 10, поэтому избегаем их на большем расстоянии
                        if dist_to_mob <= 3:
                            cost += 200  # Очень близко к призраку - очень опасно
                        elif dist_to_mob <= 6:
                            cost += 50  # В радиусе обзора призрака - опасно
                        elif dist_to_mob <= 10:
                            cost += 10  # В потенциальном радиусе преследования
                    else:
                        # Обычные мобы (patrol)
                        if dist_to_mob <= 2:
                            cost += 20  # Рядом с мобом опасно
                        elif dist_to_mob <= 3:
                            cost += 5  # Близко к мобу
                
                # Избегаем клеток с бомбами (нельзя наступить, если нет акробатики)
                bomb_at_pos = any(b.pos == neighbor for b in self.bombs)
                if bomb_at_pos and self.acrobatics_level < 1:
                    continue # Нельзя пройти
                
                new_g = g + cost
                
                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (new_g + h(neighbor), new_g, neighbor, new_path))
        
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
                min_dist = dist
                nearest = enemy
        
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
        
        return nearest
    
    def find_safe_escape_from_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[List[Position]]:
        """Найти безопасный путь отступления от призрака"""
        # Призраки могут проходить через препятствия, поэтому нужно убегать дальше
        # Ищем позицию как можно дальше от призрака в направлении от него
        
        # Вычисляем направление от призрака
        dx_dir = bomber.pos.x - ghost.pos.x
        dy_dir = bomber.pos.y - ghost.pos.y
        
        # Нормализуем направление
        if dx_dir != 0:
            dx_dir = 1 if dx_dir > 0 else -1
        if dy_dir != 0:
            dy_dir = 1 if dy_dir > 0 else -1
        
        # Пробуем несколько позиций в направлении от призрака
        best_escape = None
        max_distance = -1
        
        # Проверяем позиции на расстоянии 8-15 клеток от призрака
        for distance in range(8, 16):
            for mult_x in [-1, 0, 1]:
                for mult_y in [-1, 0, 1]:
                    if mult_x == 0 and mult_y == 0: continue
                    
                    escape_pos = Position(
                        bomber.pos.x + (dx_dir * distance * mult_x if dx_dir != 0 else distance * mult_x),
                        bomber.pos.y + (dy_dir * distance * mult_y if dy_dir != 0 else distance * mult_y)
                    )
                    
                    if not self.is_valid_position(escape_pos):
                        continue
                    
                    dist_to_ghost = escape_pos.manhattan_distance(ghost.pos)
                    
                    # Проверяем безопасность от бомб
                    if self.get_blast_danger(escape_pos) < 2.0:
                        continue
                    
                    # Проверяем путь (ограничиваем до 15 шагов для производительности)
                    escape_path = self.find_path(bomber.pos, escape_pos, max_steps=15)
                    if not escape_path or len(escape_path) < 2:
                        continue
                    
                    # Предпочитаем позиции дальше от призрака
                    if dist_to_ghost > max_distance:
                        max_distance = dist_to_ghost
                        best_escape = escape_path
                        
                        # Если нашли хорошую позицию (далеко от призрака), используем её
                        if dist_to_ghost >= 12:
                            return best_escape
        
        return best_escape
    
    def find_bomb_position_for_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить призрака"""
        # Призраки могут проходить через препятствия, но взрыв их все равно убьет
        # Ищем позицию рядом с призраком
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(ghost.pos.x + dx, ghost.pos.y + dy)
            
            if not self.is_valid_position(bomb_pos):
                continue
            
            # Проверяем, что призрак попадет в радиус взрыва
            if bomb_pos.manhattan_distance(ghost.pos) > self.bomb_range:
                continue
            
            # Проверяем безопасность размещения бомбы (агрессивный режим для запертых)
            is_enclosed = self.is_bomber_enclosed(bomber)
            can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=is_enclosed)
            if can_place and escape_path:
                return (bomb_pos, escape_path)
        
        return None
        
    def will_bomb_hit_other_bombers(self, bomb_pos: Position, bomber_id: str, planned_bombs: Optional[List[Position]] = None, planned_paths: Optional[Dict[str, List[Position]]] = None) -> bool:
        """Проверить, попадут ли другие юниты под взрыв бомбы
        
        Args:
            bomb_pos: Позиция размещения бомбы
            bomber_id: ID юнита, который ставит бомбу
            planned_bombs: Список других планируемых позиций бомб (для проверки комбинированных взрывов)
            planned_paths: Словарь планируемых путей других юнитов {bomber_id: [path]}
        """
        bomb_timer = self.bomb_delay / 1000.0
        speed = 2 + self.speed_level
        
        # Собираем все позиции бомб, которые будут взрываться одновременно
        all_bomb_positions = [bomb_pos]
        if planned_bombs:
            all_bomb_positions.extend(planned_bombs)
        
        for other_bomber in self.bombers:
            if other_bomber.id == bomber_id or not other_bomber.alive:
                continue
            
            # Проверяем текущую позицию другого юнита или его планируемую позицию
            other_pos = other_bomber.pos
            
            # Если у юнита есть планируемый путь, учитываем его конечную позицию
            if planned_paths and other_bomber.id in planned_paths:
                planned_path = planned_paths[other_bomber.id]
                if planned_path and len(planned_path) > 0:
                    # Берем последнюю позицию пути как планируемую позицию юнита
                    final_path_pos = Position(planned_path[-1][0], planned_path[-1][1])
                    # Используем планируемую позицию, если она дальше от бомбы
                    if final_path_pos.manhattan_distance(bomb_pos) > other_pos.manhattan_distance(bomb_pos):
                        other_pos = final_path_pos
            
            # Проверяем, попадет ли другой юнит в радиус взрыва ЛЮБОЙ из планируемых бомб
            in_range_of_any_bomb = False
            dangerous_bomb_pos = None
            
            for check_bomb_pos in all_bomb_positions:
                # Проверяем, попадет ли другой юнит в радиус взрыва
                # Взрыв идет по кресту (горизонталь и вертикаль)
                in_range = False
                if other_pos.y == check_bomb_pos.y and abs(other_pos.x - check_bomb_pos.x) <= self.bomb_range:
                    # Проверяем, нет ли препятствий между бомбой и юнитом
                    blocked = False
                    step = 1 if other_pos.x > check_bomb_pos.x else -1
                    for x in range(check_bomb_pos.x + step, other_pos.x, step):
                        if Position(x, check_bomb_pos.y) in self.walls or Position(x, check_bomb_pos.y) in self.obstacles:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                elif other_pos.x == check_bomb_pos.x and abs(other_pos.y - check_bomb_pos.y) <= self.bomb_range:
                    blocked = False
                    step = 1 if other_pos.y > check_bomb_pos.y else -1
                    for y in range(check_bomb_pos.y + step, other_pos.y, step):
                        if Position(check_bomb_pos.x, y) in self.walls or Position(check_bomb_pos.x, y) in self.obstacles:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                
                if in_range:
                    in_range_of_any_bomb = True
                    dangerous_bomb_pos = check_bomb_pos
                    break
            
            if in_range_of_any_bomb:
                # КРИТИЧНО: Если другой юнит в радиусе взрыва, проверяем, успеет ли он убежать
                # Используем более гибкую проверку - достаточно, чтобы юнит мог отойти хотя бы на 1 шаг
                safe_escape_found = False
                
                # Проверяем все соседние позиции для escape
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    escape_pos = Position(other_pos.x + dx, other_pos.y + dy)
                    if not self.is_valid_position(escape_pos):
                        continue
                    
                    # Проверяем, что escape позиция не в радиусе взрыва ЛЮБОЙ из планируемых бомб
                    escape_in_blast = False
                    for check_bomb_pos in all_bomb_positions:
                        if escape_pos.x == check_bomb_pos.x and abs(escape_pos.y - check_bomb_pos.y) <= self.bomb_range:
                            escape_in_blast = True
                            break
                        elif escape_pos.y == check_bomb_pos.y and abs(escape_pos.x - check_bomb_pos.x) <= self.bomb_range:
                            escape_in_blast = True
                            break
                    
                    if escape_in_blast:
                        continue
                    
                    # Проверяем, что escape позиция безопасна от других бомб (уже установленных)
                    danger = self.get_blast_danger(escape_pos)
                    if danger < 1.0:  # Должна быть безопасна минимум 1 секунду
                        continue
                    
                    # Оцениваем время до escape (1 шаг = 1/speed секунд)
                    dist = other_pos.manhattan_distance(escape_pos)
                    time_to_escape = dist / speed
                    
                    # Требуем достаточно времени для безопасности (минимум 1.5 секунды запас)
                    # Это гарантирует, что юнит успеет сделать хотя бы 1 шаг
                    if time_to_escape < bomb_timer - 1.5:
                        # Дополнительно проверяем, что escape позиция действительно безопасна
                        # (не в радиусе взрыва и не опасна от других бомб)
                        escape_danger = self.get_blast_danger(escape_pos, time_offset=time_to_escape)
                        if escape_danger > bomb_timer + 0.5:  # Должна быть безопасна минимум 0.5 секунды после взрыва
                            # Проверяем, что escape позиция не занята другими юнитами
                            pos_occupied = False
                            for other_b in self.bombers:
                                if other_b.id != other_bomber.id and other_b.alive and other_b.pos == escape_pos:
                                    pos_occupied = True
                                    break
                            
                            if not pos_occupied:
                                safe_escape_found = True
                                break
                
                # Если другой юнит не может убежать - ЗАПРЕЩАЕМ ставить бомбу
                if not safe_escape_found:
                    print(f"[BOMB SAFETY] {bomber_id} cannot place bomb at {bomb_pos} - would hit {other_bomber.id} at {other_pos} (no safe escape)")
                    return True
        
        return False
    
    def can_safely_place_bomb(self, bomber: Bomber, bomb_pos: Position, aggressive: bool = False) -> Tuple[bool, Optional[List[Position]]]:
        """Проверить, можно ли безопасно поставить бомбу и убежать (с учетом других юнитов)
        
        Args:
            bomber: Юнит, который хочет поставить бомбу
            bomb_pos: Позиция для размещения бомбы
            aggressive: Если True, более мягкие требования безопасности (для запертых юнитов)
        """
        if bomber.bombs_available == 0:
            return False, None
        
        # Проверяем, не попадут ли другие юниты под взрыв
        if self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
            return False, None
        
        # Симулируем размещение бомбы
        bomb_timer = self.bomb_delay / 1000.0
        min_safe_distance = self.bomb_range + 2
        speed = 2 + self.speed_level
        
        # КРИТИЧНО: Проверяем, что текущая позиция юнита не попадет под взрыв
        # (если юнит стоит на позиции бомбы или рядом, он должен убежать)
        bomber_in_blast = False
        if bomber.pos.x == bomb_pos.x and abs(bomber.pos.y - bomb_pos.y) <= self.bomb_range:
            # Проверяем, нет ли препятствий между бомбой и юнитом
            blocked = False
            step = 1 if bomber.pos.y > bomb_pos.y else -1
            for y in range(bomb_pos.y + step, bomber.pos.y, step):
                if Position(bomb_pos.x, y) in self.walls:
                    blocked = True
                    break
            if not blocked:
                bomber_in_blast = True
        elif bomber.pos.y == bomb_pos.y and abs(bomber.pos.x - bomb_pos.x) <= self.bomb_range:
            blocked = False
            step = 1 if bomber.pos.x > bomb_pos.x else -1
            for x in range(bomb_pos.x + step, bomber.pos.x, step):
                if Position(x, bomb_pos.y) in self.walls:
                    blocked = True
                    break
            if not blocked:
                bomber_in_blast = True
        
        # Если юнит попадет под взрыв, он ОБЯЗАТЕЛЬНО должен убежать
        if bomber_in_blast:
            # Для агрессивного режима используем более мягкие требования, но все равно требуем escape
            safety_margin = 0.3 if aggressive else 0.5
            min_danger_time = 0.2 if aggressive else 0.5
        else:
            # Для агрессивного режима снижаем требования
            safety_margin = 0.5 if aggressive else 1.0
            min_danger_time = 0.3 if aggressive else 1.0
        
        # Проверяем соседние клетки
        safe_escape_positions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            escape_pos = Position(bomb_pos.x + dx, bomb_pos.y + dy)
            if not self.is_valid_position(escape_pos):
                continue
                
            # Проверяем, что escape позиция не в радиусе взрыва
            # Используем точную проверку по кресту (взрыв идет по горизонтали и вертикали)
            in_blast_radius = False
            if escape_pos.x == bomb_pos.x and abs(escape_pos.y - bomb_pos.y) <= self.bomb_range:
                # Проверяем, нет ли препятствий между бомбой и escape позицией
                blocked = False
                step = 1 if escape_pos.y > bomb_pos.y else -1
                for y in range(bomb_pos.y + step, escape_pos.y, step):
                    if Position(bomb_pos.x, y) in self.walls:
                        blocked = True
                        break
                if not blocked:
                    in_blast_radius = True
            elif escape_pos.y == bomb_pos.y and abs(escape_pos.x - bomb_pos.x) <= self.bomb_range:
                blocked = False
                step = 1 if escape_pos.x > bomb_pos.x else -1
                for x in range(bomb_pos.x + step, escape_pos.x, step):
                    if Position(x, bomb_pos.y) in self.walls:
                        blocked = True
                        break
                if not blocked:
                    in_blast_radius = True
            
            if in_blast_radius:
                continue
                
            # Проверяем путь к escape позиции
            escape_path = self.find_path(bomber.pos, escape_pos, max_steps=15)
            if not escape_path or len(escape_path) < 2:
                # Для агрессивного режима пробуем прямую соседнюю клетку
                if aggressive and bomber.pos.manhattan_distance(escape_pos) == 1:
                    escape_path = [bomber.pos, escape_pos]
                else:
                    continue
                
            # Оцениваем время до достижения escape позиции
            time_to_escape = len(escape_path) / speed
            
            # Проверяем, что успеем убежать до взрыва
            if time_to_escape < bomb_timer - safety_margin:
                # Проверяем, что escape позиция безопасна от других бомб
                danger_time = self.get_blast_danger(escape_pos, time_offset=time_to_escape)
                if danger_time > min_danger_time:
                    safe_escape_positions.append((escape_pos, escape_path, time_to_escape))
        
        if safe_escape_positions:
            # Выбираем ближайшую безопасную позицию
            best_escape = min(safe_escape_positions, key=lambda x: len(x[1]))
            return True, best_escape[1]
        
        return False, None
    
    def find_bomb_position_for_enemy(self, bomber: Bomber, enemy: Enemy) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить врага, и путь отступления"""
        # Ищем позицию рядом с врагом, где можно поставить бомбу
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(enemy.pos.x + dx, enemy.pos.y + dy)
                
            if not self.is_valid_position(bomb_pos):
                    continue
                
            # Проверяем, что враг попадет в радиус взрыва
            if bomb_pos.manhattan_distance(enemy.pos) > self.bomb_range:
                continue
            
            # Проверяем безопасность размещения бомбы (агрессивный режим для запертых)
            is_enclosed = self.is_bomber_enclosed(bomber)
            can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=is_enclosed)
            if can_place and escape_path:
                return (bomb_pos, escape_path)
        
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
                if bomb_pos == obs and self.acrobatics_level < 2:
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
            vision_range = 5 + (3 * (self.speed_level if hasattr(self, 'speed_level') else 0))
            
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
            speed = 2 + self.speed_level  # клетки/сек
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
        """Основная логика хода - агрессивная стратегия Bomberman"""
        commands = []
        assigned_targets = set()  # Отслеживаем назначенные цели для координации
        
        # Первый проход: собираем информацию о целях всех юнитов
        for bomber_idx, bomber in enumerate(self.bombers):
            if not bomber.alive or not bomber.can_move:
                print(f"[BOMBER] {bomber.id} at {bomber.pos} - not alive or can't move")
                continue
            print(f"[BOMBER] Processing {bomber.id} (#{bomber_idx}) at {bomber.pos}, bombs={bomber.bombs_available}")
            
            # Пропускаем критичные действия (escape), но собираем цели для атаки
            danger_timer = self.get_blast_danger(bomber.pos)
            if danger_timer >= 1.5:  # Не в критической опасности
                # Приоритет: призраки опаснее обычных врагов
                nearby_ghost = self.find_nearby_ghost(bomber, max_distance=10)
                if nearby_ghost:
                    assigned_targets.add(nearby_ghost.pos)
                else:
                    nearby_enemy = self.find_nearby_enemy(bomber, max_distance=4)
                    if nearby_enemy:
                        assigned_targets.add(nearby_enemy.pos)
                    else:
                        # Предварительно оцениваем цели
                        target = self.get_best_target(bomber, assigned_targets)
                        if target:
                            assigned_targets.add(target)
        
        # Собираем все планируемые позиции бомб для проверки безопасности
        planned_bomb_positions = {}  # bomber_id -> bomb_pos
        
        # Второй проход: выполняем действия с учетом координации
        for bomber_idx, bomber in enumerate(self.bombers):
            if not bomber.alive or not bomber.can_move: continue
            
            my_commands = {"id": bomber.id, "path": [], "bombs": []}
            
            # ПРИОРИТЕТ 0: УБЕЖАТЬ ОТ ВЗРЫВА (КРИТИЧНО!)
            # Проверяем опасность от ВСЕХ бомб (не только своих)
            danger_timer = self.get_blast_danger(bomber.pos)
            if danger_timer < 3.0: # Если взрыв через < 3 сек - СРОЧНО УБЕГАТЬ!
                print(f"[BOMB ESCAPE] {bomber.id} at {bomber.pos} in danger! Timer={danger_timer:.2f}")
                
                # Используем продвинутую логику для поиска Sanctuary, если доступна
                if self.use_advanced_logic and self.safe_pathfinder:
                    sanctuary = self.safe_pathfinder.find_sanctuary((bomber.pos.x, bomber.pos.y), max_ticks=30)
                    if sanctuary:
                        sanctuary_pos = Position(sanctuary[0], sanctuary[1])
                        # Ищем путь к безопасной зоне
                        path_to_sanctuary = self.find_path(bomber.pos, sanctuary_pos, max_steps=20)
                        if path_to_sanctuary and len(path_to_sanctuary) > 1:
                            my_commands["path"] = [[p.x, p.y] for p in path_to_sanctuary[:15]]
                            commands.append(my_commands)
                            print(f"[BOMB ESCAPE] {bomber.id} escaping to sanctuary at {sanctuary_pos} (advanced logic)")
                            continue
                
                # Fallback на старую логику
                best_escape = None
                max_timer = -1
                
                # Получаем позиции других юнитов для избежания столкновений
                other_bomber_positions = {b.pos for b in self.bombers if b.id != bomber.id and b.alive}
                
                # Проверяем все соседние позиции и ищем самую безопасную
                escape_candidates = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    esc_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if not self.is_valid_position(esc_pos):
                        continue
                    
                    # Избегаем позиций, где уже есть другие юниты
                    if esc_pos in other_bomber_positions:
                        continue
                    
                    # Проверяем опасность от всех бомб
                    t = self.get_blast_danger(esc_pos)
                    if t > max_timer:
                        max_timer = t
                        best_escape = esc_pos
                    escape_candidates.append((esc_pos, t))
                
                # Если нашли безопасную позицию, убегаем туда
                if best_escape and max_timer > danger_timer + 0.5:
                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape.x, best_escape.y]]
                    commands.append(my_commands)
                    print(f"[BOMB ESCAPE] {bomber.id} escaping to {best_escape} (safety={max_timer:.2f})")
                    continue
                elif escape_candidates:
                    # Если нет идеально безопасной позиции, выбираем самую безопасную из доступных
                    escape_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_escape, best_safety = escape_candidates[0]
                    if best_safety > danger_timer:
                        my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape.x, best_escape.y]]
                        commands.append(my_commands)
                        print(f"[BOMB ESCAPE] {bomber.id} emergency escape to {best_escape} (safety={best_safety:.2f})")
                        continue
            
            # ПРИОРИТЕТ 1: УБЕЖАТЬ ОТ ПРИЗРАКА (КРИТИЧНО! Призраки могут проходить через препятствия)
            nearby_ghost = self.find_nearby_ghost(bomber, max_distance=12)
            if nearby_ghost:
                # Призрак в радиусе обзора (10 клеток) - СРОЧНО УБЕГАТЬ!
                # Призраки могут проходить через препятствия, поэтому убегаем дальше
                escape_path = self.find_safe_escape_from_ghost(bomber, nearby_ghost)
                if escape_path and len(escape_path) > 1:
                    my_commands["path"] = [[p.x, p.y] for p in escape_path[:12]]
                    commands.append(my_commands)
                    continue
                else:
                    # Если не нашли путь, пробуем просто убежать в любую сторону (более агрессивно)
                    other_bomber_positions = {b.pos for b in self.bombers if b.id != bomber.id and b.alive}
                    best_escape = None
                    best_score = -1
                    
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                        if not self.is_valid_position(escape_pos):
                            continue
                        if escape_pos in other_bomber_positions:
                            continue
                        
                        # Считаем расстояние от призрака и безопасность
                        dist_from_ghost = escape_pos.manhattan_distance(nearby_ghost.pos)
                        danger = self.get_blast_danger(escape_pos)
                        
                        # Приоритет: расстояние от призрака > безопасность (но безопасность должна быть > 0.5)
                        score = dist_from_ghost * 10 + danger
                        if danger > 0.5 and score > best_score:  # Снизили требование безопасности
                            best_score = score
                            best_escape = escape_pos
                    
                    if best_escape:
                        my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape.x, best_escape.y]]
                        commands.append(my_commands)
                        continue
                
            # ПРИОРИТЕТ 2.8: СПЕЦИАЛЬНОЕ УБЕГАНИЕ ОТ ПРИЗРАКОВ (они опаснее!)
            nearby_ghost = self.find_nearby_ghost(bomber, max_distance=8)  # Призраки опасны на большем расстоянии
            if nearby_ghost:
                dist_to_ghost = bomber.pos.manhattan_distance(nearby_ghost.pos)
                print(f"[GHOST ESCAPE] {bomber.id} at {bomber.pos} escaping from ghost at {nearby_ghost.pos}, distance={dist_to_ghost}")

                # РЕЖИМ ПАНИКИ: если призрак ОЧЕНЬ близко (<= 4 клетки)
                if dist_to_ghost <= 4:
                    print(f"[PANIC] {bomber.id} PANIC MODE! Ghost too close!")
                    # Бежим в любом направлении от призрака
                    dx = bomber.pos.x - nearby_ghost.pos.x
                    dy = bomber.pos.y - nearby_ghost.pos.y

                    # Нормализуем направление
                    if abs(dx) > abs(dy):
                        move_dx, move_dy = (1 if dx > 0 else -1), 0
                    else:
                        move_dx, move_dy = 0, (1 if dy > 0 else -1)

                    # Пробуем несколько позиций в направлении от призрака
                    for dist in range(1, 6):
                        escape_pos = Position(
                            bomber.pos.x + move_dx * dist,
                            bomber.pos.y + move_dy * dist
                        )
                        if self.is_valid_position(escape_pos) and self.get_blast_danger(escape_pos) > 1.0:
                            print(f"[PANIC] {bomber.id} emergency escape to {escape_pos}")
                            my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                            commands.append(my_commands)
                            break
                    else:
                        # Если не нашли хорошую позицию, бежим в любую сторону
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                            if self.is_valid_position(escape_pos):
                                print(f"[PANIC] {bomber.id} desperate escape to {escape_pos}")
                                my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                                commands.append(my_commands)
                                break
                    continue

                # Для не очень близких призраков используем специальную логику убегания
                escape_path = self.find_safe_escape_from_ghost(bomber, nearby_ghost)
                if escape_path and len(escape_path) > 1:
                    print(f"[GHOST ESCAPE] {bomber.id} using safe escape path: {len(escape_path)} steps")
                    my_commands["path"] = [[p.x, p.y] for p in escape_path[:12]]
                    commands.append(my_commands)
                    continue
                else:
                    # Если нет безопасного пути, используем обычную логику убегания
                    print(f"[GHOST ESCAPE] {bomber.id} no safe escape path, using emergency escape")

            # ПРИОРИТЕТ 2: АТАКОВАТЬ ПРИЗРАКА БОМБОЙ (если он близко и есть бомбы)
            nearby_ghost = self.find_nearby_ghost(bomber, max_distance=6)
            if nearby_ghost and bomber.bombs_available > 0:
                dist_to_ghost = bomber.pos.manhattan_distance(nearby_ghost.pos)
                print(f"[GHOST ATTACK] {bomber.id} attacking ghost at {nearby_ghost.pos}, distance={dist_to_ghost}")
                bomb_result = self.find_bomb_position_for_ghost(bomber, nearby_ghost)
                if bomb_result:
                    bomb_pos, escape_path = bomb_result
                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=4)
                    if path_to_bomb and len(path_to_bomb) > 1:
                        if len(path_to_bomb) <= 2:
                            # Бомба ставится на текущей позиции или следующей
                            bomb_placement_pos = path_to_bomb[-1] if len(path_to_bomb) > 1 else bomber.pos
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in escape_path[:10]]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                print(f"[GHOST ATTACK] {bomber.id} placing bomb at {bomb_placement_pos}")
                            else:
                                print(f"[GHOST ATTACK] {bomber.id} cannot place bomb at {bomb_placement_pos} - would hit other bombers")
                                continue
                        else:
                            # Бомба ставится на последней позиции пути
                            bomb_placement_pos = path_to_bomb[-1]
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            else:
                                print(f"[GHOST ATTACK] {bomber.id} cannot place bomb at {bomb_placement_pos} - would hit other bombers")
                                continue
                        commands.append(my_commands)
                        continue
            
            # ПРИОРИТЕТ 3: АТАКОВАТЬ ВРАГА БОМБОЙ (высокий приоритет!)
            nearby_enemy = self.find_nearby_enemy(bomber, max_distance=8)  # Увеличили радиус поиска
            if nearby_enemy and bomber.bombs_available > 0:
                enemy_dist = bomber.pos.manhattan_distance(nearby_enemy.pos)
                print(f"[ENEMY ATTACK] {bomber.id} attacking enemy at {nearby_enemy.pos}, distance={enemy_dist}")
                
                bomb_result = self.find_bomb_position_for_enemy(bomber, nearby_enemy)
                if bomb_result:
                    bomb_pos, escape_path = bomb_result
                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=6)
                    if path_to_bomb and len(path_to_bomb) > 1:
                        if len(path_to_bomb) <= 2:
                            # Бомба ставится на текущей позиции или следующей
                            bomb_placement_pos = path_to_bomb[-1] if len(path_to_bomb) > 1 else bomber.pos
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in escape_path[:8]]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                print(f"[ENEMY ATTACK] {bomber.id} placing bomb at {bomb_placement_pos} to kill enemy")
                            else:
                                print(f"[ENEMY ATTACK] {bomber.id} cannot place bomb at {bomb_placement_pos} - would hit other bombers")
                                continue
                        else:
                            # Бомба ставится на последней позиции пути
                            bomb_placement_pos = path_to_bomb[-1]
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                print(f"[ENEMY ATTACK] {bomber.id} moving to {bomb_pos} to attack enemy")
                            else:
                                print(f"[ENEMY ATTACK] {bomber.id} cannot place bomb at {bomb_placement_pos} - would hit other bombers")
                                continue
                        commands.append(my_commands)
                        continue
                
                # Если враг очень близко (<= 3 клетки), пробуем поставить бомбу агрессивнее
                if enemy_dist <= 3:
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
                            is_enclosed = self.is_bomber_enclosed(bomber)
                            can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=True)
                            if can_place and escape_path:
                                if bomber.pos == bomb_pos:
                                    # Проверяем, не попадут ли другие юниты под взрыв
                                    if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                                        # Бомба ставится на текущей позиции
                                        my_commands["path"] = [[p.x, p.y] for p in escape_path[:8]]
                                        my_commands["bombs"] = [[bomb_pos.x, bomb_pos.y]]
                                        print(f"[ENEMY ATTACK] {bomber.id} aggressive bomb at {bomb_pos} (current pos)")
                                        commands.append(my_commands)
                                        break
                                else:
                                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=4)
                                    if path_to_bomb and len(path_to_bomb) > 1:
                                        # Проверяем, не попадут ли другие юниты под взрыв
                                        bomb_placement_pos = path_to_bomb[-1]
                                        if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                            # Бомба ставится на последней позиции пути
                                            my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                            my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                            commands.append(my_commands)
                                            break
                    # Если нашли безопасное размещение бомбы, выходим из цикла
                    if my_commands.get("bombs"):
                        continue
            
            # ПРИОРИТЕТ 3.5: УБЕЖАТЬ ОТ ВРАГА/МОБА (если очень близко и нет бомб)
            nearby_enemy = self.find_nearby_enemy(bomber, max_distance=3)
            if nearby_enemy:
                is_ghost = hasattr(nearby_enemy, 'type') and nearby_enemy.type == "ghost"
                enemy_type = "ghost" if is_ghost else "enemy/mob"
                dist_to_enemy = bomber.pos.manhattan_distance(nearby_enemy.pos)
                print(f"[ESCAPE] {bomber.id} escaping from {enemy_type} at {nearby_enemy.pos}, distance={dist_to_enemy}")
                # Враг очень близко - СРОЧНО УБЕГАТЬ!
                other_bomber_positions = {b.pos for b in self.bombers if b.id != bomber.id and b.alive}
                
                # Сначала пробуем в любую безопасную сторону (более агрессивно)
                best_escape = None
                best_score = -1
                best_escape_occupied = None  # Лучшая позиция, даже если занята другим юнитом
                best_score_occupied = -1
                
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if not self.is_valid_position(escape_pos):
                        continue
                    
                    # Считаем расстояние от врага (чем дальше - тем лучше)
                    dist_from_enemy = escape_pos.manhattan_distance(nearby_enemy.pos)
                    danger = self.get_blast_danger(escape_pos)
                    
                    # Приоритет: расстояние от врага > безопасность (но безопасность должна быть > 0.5)
                    score = dist_from_enemy * 10 + danger
                    
                    if escape_pos in other_bomber_positions:
                        # Позиция занята другим юнитом, но сохраняем как запасной вариант
                        if danger > 0.3 and score > best_score_occupied:  # Еще более низкие требования
                            best_score_occupied = score
                            best_escape_occupied = escape_pos
                        continue

                    if danger > 0.5 and score > best_score:  # Снизили требование безопасности
                        best_score = score
                        best_escape = escape_pos
                
                # Используем лучшую свободную позицию
                if best_escape:
                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape.x, best_escape.y]]
                    commands.append(my_commands)
                    continue
                
                # Если нет свободных позиций, но есть занятая - используем её (лучше чем стоять)
                if best_escape_occupied:
                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape_occupied.x, best_escape_occupied.y]]
                    commands.append(my_commands)
                    continue
                
                # Если не нашли безопасную позицию, пробуем путь отступления
                escape_direction = Position(
                    bomber.pos.x + (bomber.pos.x - nearby_enemy.pos.x),
                    bomber.pos.y + (bomber.pos.y - nearby_enemy.pos.y)
                )
                escape_path = self.find_path(bomber.pos, escape_direction, max_steps=10)
                
                if escape_path and len(escape_path) > 1:
                    my_commands["path"] = [[p.x, p.y] for p in escape_path[:5]]
                    commands.append(my_commands)
                    continue
            
            
            # ПРИОРИТЕТ 3.5: ПРОВЕРКА НА ЗАПЕРТОСТЬ (высокий приоритет для запертых юнитов)
            is_enclosed = self.is_bomber_enclosed(bomber)
            if is_enclosed:
                print(f"[ENCLOSED] {bomber.id} is enclosed, prioritizing escape/breakthrough")
                if self.attempt_enclosure_escape(bomber, my_commands, commands):
                    continue
            
            # ПРИОРИТЕТ 4: УНИЧТОЖЕНИЕ ПРЕПЯТСТВИЙ БОМБАМИ (БАЗОВАЯ ЛОГИКА)
            if bomber.bombs_available > 0:
                # ПРОСТАЯ ЛОГИКА: ищем препятствия рядом и ставим бомбы
                # Исключаем препятствия, которые уже назначены другим бомберам
                nearby_obstacles = []
                bomber_idx = next((i for i, b in enumerate(self.bombers) if b.id == bomber.id), 0)
                
                for obs in self.obstacles:
                    # Пропускаем препятствия, которые уже назначены другим бомберам
                    if obs in assigned_targets:
                        continue
                    dist = bomber.pos.manhattan_distance(obs)
                    # Ищем препятствия в радиусе взрыва + 3 клетки (расширили радиус)
                    if 1 <= dist <= self.bomb_range + 3:
                        # Добавляем небольшой бонус для разных бомберов, чтобы они выбирали разные препятствия
                        # Используем индекс бомбера для распределения препятствий
                        priority_bonus = (hash(obs) + bomber_idx) % 10  # Небольшой бонус для распределения
                        nearby_obstacles.append((obs, dist - priority_bonus * 0.1))
                
                if nearby_obstacles:
                    # Сортируем по близости
                    nearby_obstacles.sort(key=lambda x: x[1])
                    
                    # Получаем позиции других бомберов для избежания столкновений
                    other_bomber_positions = {b.pos for b in self.bombers if b.id != bomber.id and b.alive}
                    
                    # Пробуем найти позицию для бомбы рядом с ближайшим препятствием
                    escape_found = False
                    for obs, dist in nearby_obstacles[:5]:  # Проверяем топ-5 ближайших
                        # Пробуем поставить бомбу рядом с препятствием
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            bomb_pos = Position(obs.x + dx, obs.y + dy)
                            
                            if not self.is_valid_position(bomb_pos):
                                continue
                            
                            # Проверяем, попадет ли препятствие в радиус взрыва (по кресту)
                            in_range = False
                            # Взрыв идет по кресту: горизонталь и вертикаль
                            if bomb_pos.x == obs.x:
                                # Одна вертикальная линия - проверяем расстояние по Y
                                y_dist = abs(bomb_pos.y - obs.y)
                                if y_dist <= self.bomb_range:
                                    # Проверяем, нет ли стен между бомбой и препятствием
                                    blocked = False
                                    step = 1 if obs.y > bomb_pos.y else -1
                                    for y in range(bomb_pos.y + step, obs.y, step):
                                        if Position(bomb_pos.x, y) in self.walls:
                                            blocked = True
                                            break
                                    if not blocked:
                                        in_range = True
                            elif bomb_pos.y == obs.y:
                                # Одна горизонтальная линия - проверяем расстояние по X
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
                            
                            if not in_range:
                                continue
                            
                            # Если мы уже на позиции бомбы - ставим бомбу
                            if bomber.pos == bomb_pos:
                                # Ищем escape позицию
                                escape_candidates = []
                                for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                                    if not self.is_valid_position(escape_pos):
                                        continue
                                    
                                    # Пропускаем позиции, занятые другими бомберами
                                    if escape_pos in other_bomber_positions:
                                        continue
                                    
                                    # Проверяем, что escape позиция не в радиусе взрыва
                                    escape_in_blast = False
                                    if escape_pos.x == bomb_pos.x and abs(escape_pos.y - bomb_pos.y) <= self.bomb_range:
                                        escape_in_blast = True
                                    elif escape_pos.y == bomb_pos.y and abs(escape_pos.x - bomb_pos.x) <= self.bomb_range:
                                        escape_in_blast = True
                                    
                                    if not escape_in_blast:
                                        # Проверяем базовую безопасность от других бомб
                                        danger = self.get_blast_danger(escape_pos)
                                        escape_candidates.append((escape_pos, danger))
                                
                                # Выбираем лучшую escape позицию
                                if escape_candidates:
                                    escape_candidates.sort(key=lambda x: x[1], reverse=True)
                                    escape_pos, danger = escape_candidates[0]
                                    
                                    if danger > 0.2:
                                        # Проверяем, не попадут ли другие юниты под взрыв
                                        if not self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
                                            assigned_targets.add(obs)
                                            # Бомба ставится на текущей позиции
                                            my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                                            my_commands["bombs"] = [[bomb_pos.x, bomb_pos.y]]
                                            commands.append(my_commands)
                                            print(f"[OBSTACLE] {bomber.id} placing bomb at {bomb_pos} (current pos) to destroy obstacle at {obs}")
                                            escape_found = True
                                            break
                            else:
                                # Идем к позиции бомбы (бомба будет поставлена на пути)
                                if bomb_pos not in other_bomber_positions:
                                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=5)
                                    if path_to_bomb and len(path_to_bomb) > 1:
                                        bomb_placement_pos = path_to_bomb[-1]
                                        # Проверяем, не попадут ли другие юниты под взрыв
                                        if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                            assigned_targets.add(obs)
                                            # Бомба ставится на последней позиции пути
                                            my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                            my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                            commands.append(my_commands)
                                            print(f"[OBSTACLE] {bomber.id} moving to {bomb_pos} (path len={len(path_to_bomb)}) to destroy obstacle at {obs}")
                                            escape_found = True
                                            break
                        
                        # Если нашли позицию для бомбы, выходим из цикла по препятствиям
                        if escape_found:
                            break
                    
                    # Если нашли действие, продолжаем
                    if escape_found:
                        continue
                
                # Дополнительно: используем улучшенную функцию прорыва
                aggressive = is_enclosed
                breakthrough = self.find_bomb_position_for_obstacle_breakthrough(bomber, aggressive=aggressive)
                if breakthrough:
                    bomb_pos, escape_path = breakthrough
                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=5)
                    if path_to_bomb and len(path_to_bomb) > 1:
                        if len(path_to_bomb) <= 2:
                            # Бомба ставится на текущей позиции или следующей
                            bomb_placement_pos = path_to_bomb[-1] if len(path_to_bomb) > 1 else bomber.pos
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in escape_path[:8]]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            else:
                                continue
                        else:
                            # Бомба ставится на последней позиции пути
                            bomb_placement_pos = path_to_bomb[-1]
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            else:
                                continue
                        commands.append(my_commands)
                        continue
            
            # ПРИОРИТЕТ 4.5: БОМБОВЫЙ ПОЕЗД - координация для цепных взрывов
            if bomber.bombs_available > 0 and len(self.bombers) >= 2:
                chain_bomb_target = self.find_chain_bomb_opportunity(bomber, bomber_idx, assigned_targets)
                if chain_bomb_target:
                    bomb_pos, escape_path = chain_bomb_target
                    path_to_bomb = self.find_path(bomber.pos, bomb_pos, max_steps=6)
                    if path_to_bomb and len(path_to_bomb) > 1:
                        if len(path_to_bomb) <= 2:
                            # Бомба ставится на текущей позиции или следующей
                            bomb_placement_pos = path_to_bomb[-1] if len(path_to_bomb) > 1 else bomber.pos
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in escape_path[:10]]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            else:
                                continue
                        else:
                            # Бомба ставится на последней позиции пути
                            bomb_placement_pos = path_to_bomb[-1]
                            # Проверяем, не попадут ли другие юниты под взрыв
                            if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                my_commands["path"] = [[p.x, p.y] for p in path_to_bomb]
                                my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                            else:
                                continue
                        commands.append(my_commands)
                        continue

            # ПРИОРИТЕТ 5: АТАКА ПРЕПЯТСТВИЙ ДЛЯ ОЧКОВ (с координацией)
            target = self.get_best_target(bomber, assigned_targets, bomber_idx)
            if target:
                # Помечаем цель как назначенную
                assigned_targets.add(target)
                target_is_obstacle = target in self.obstacles
                move_target = target
                
                if target_is_obstacle:
                    # Ищем соседнюю клетку для размещения бомбы
                    min_dist = float('inf')
                    best_bomb_pos = None
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        p = Position(target.x + dx, target.y + dy)
                        if self.is_valid_position(p):
                            d = bomber.pos.manhattan_distance(p)
                            if d < min_dist:
                                min_dist = d
                                move_target = p
                                best_bomb_pos = p
                    
                    # Если мы уже рядом и можем поставить бомбу
                    if best_bomb_pos and bomber.bombs_available > 0:
                        dist_to_bomb = bomber.pos.manhattan_distance(best_bomb_pos)
                        if dist_to_bomb <= 2:
                            is_enclosed = self.is_bomber_enclosed(bomber)
                            can_place, escape_path = self.can_safely_place_bomb(bomber, best_bomb_pos, aggressive=is_enclosed)
                            if can_place and escape_path:
                                if bomber.pos == best_bomb_pos:
                                    # Проверяем, не попадут ли другие юниты под взрыв
                                    if not self.will_bomb_hit_other_bombers(best_bomb_pos, bomber.id):
                                        # Бомба ставится на текущей позиции
                                        my_commands["path"] = [[p.x, p.y] for p in escape_path[:8]]
                                        my_commands["bombs"] = [[best_bomb_pos.x, best_bomb_pos.y]]
                                        commands.append(my_commands)
                                        continue
                                else:
                                    # Идем к позиции бомбы, бомба ставится на последней позиции пути
                                    path = self.find_path(bomber.pos, best_bomb_pos, max_steps=3)
                                    if path and len(path) > 1:
                                        bomb_placement_pos = path[-1]
                                        # Проверяем, не попадут ли другие юниты под взрыв
                                        if not self.will_bomb_hit_other_bombers(bomb_placement_pos, bomber.id):
                                            my_commands["path"] = [[p.x, p.y] for p in path]
                                            my_commands["bombs"] = [[bomb_placement_pos.x, bomb_placement_pos.y]]
                                            commands.append(my_commands)
                                            continue
                
                # Обычное движение к цели
                path = self.find_path(bomber.pos, move_target, max_steps=10)
                if path and len(path) > 1:
                    # Проверяем, что путь безопасен от бомб
                    path_safe = True
                    for i, path_pos in enumerate(path[1:], 1):  # Пропускаем первую позицию (текущая)
                        # Время прибытия на эту позицию
                        arrival_time = i * 0.3  # Предполагаем скорость ~3 клетки/сек
                        danger = self.get_blast_danger(path_pos, time_offset=arrival_time)
                        if danger < arrival_time + 1.0:  # Бомба взорвется пока мы там
                            path_safe = False
                            break
                    
                    if path_safe:
                        my_commands["path"] = [[p.x, p.y] for p in path[:8]]
                        commands.append(my_commands)
                        continue
            
            # Этот блок уже обработан выше (ПРИОРИТЕТ 3.5), но оставляем как запасной вариант

            # ПРИОРИТЕТ 6: RANDOM ROAM (чтобы не стоять)
            # Снижаем требования к безопасности - лучше двигаться, чем стоять
            valid_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                p = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                if self.is_valid_position(p):
                    danger = self.get_blast_danger(p)
                    # Принимаем позиции с опасностью > 0.5 (было 2.0)
                    if danger > 0.5:
                        valid_moves.append((p, danger))
            
            if valid_moves:
                # Выбираем самую безопасную из доступных
                valid_moves.sort(key=lambda x: x[1], reverse=True)
                chosen = valid_moves[0][0]
                my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [chosen.x, chosen.y]]
                commands.append(my_commands)
            else:
                # Если вообще нет безопасных позиций, все равно двигаемся (лучше чем стоять)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    p = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if self.is_valid_position(p):
                        my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [p.x, p.y]]
                        commands.append(my_commands)
                        break

        # Валидация и финальная проверка безопасности: проверяем все пути и бомбы
        if commands:
            # Валидируем все пути перед отправкой и фильтруем мертвых юнитов
            validated_commands = []
            for cmd in commands:
                bomber_id = cmd["id"]
                bomber = next((b for b in self.bombers if b.id == bomber_id), None)
                if not bomber:
                    print(f"[COMMAND FILTER] Bomber {bomber_id} not found, skipping command")
                    continue
                
                # Фильтруем мертвых юнитов
                if not bomber.alive:
                    print(f"[COMMAND FILTER] Bomber {bomber_id} is dead, skipping command")
                    continue
                
                # Фильтруем юнитов, которые не могут двигаться
                if not bomber.can_move and cmd.get("path"):
                    print(f"[COMMAND FILTER] Bomber {bomber_id} cannot move, removing path")
                    cmd["path"] = []
                
                # Валидируем путь
                if cmd.get("path"):
                    valid_path = self._validate_path(cmd["path"], bomber.pos)
                    if valid_path:
                        cmd["path"] = valid_path
                    else:
                        print(f"[PATH VALIDATION] Removing invalid path for {bomber_id}")
                        cmd["path"] = []
                
                # Проверяем, что путь не пустой или есть бомба
                if cmd.get("path") or cmd.get("bombs"):
                    validated_commands.append(cmd)
                else:
                    print(f"[PATH VALIDATION] Removing empty command for {bomber_id}")
            
            commands = validated_commands
        
        # Финальная проверка безопасности: проверяем все бомбы с учетом всех других планируемых бомб
        if commands:
            # Собираем все планируемые позиции бомб, позиции юнитов и пути
            all_planned_bombs = []
            bomber_positions = {}  # bomber_id -> Position
            planned_paths = {}  # bomber_id -> path
            for cmd in commands:
                bomber_id = cmd["id"]
                # Находим позицию юнита
                bomber = next((b for b in self.bombers if b.id == bomber_id), None)
                if bomber:
                    bomber_positions[bomber_id] = bomber.pos
                
                # Сохраняем планируемые пути
                if cmd.get("path"):
                    planned_paths[bomber_id] = cmd["path"]
                
                if cmd.get("bombs"):
                    bomb_pos = Position(cmd["bombs"][0][0], cmd["bombs"][0][1])
                    all_planned_bombs.append(bomb_pos)
            
            # Дополнительная проверка: если несколько юнитов находятся рядом (в радиусе 2 клеток)
            # и все ставят бомбы, это опасно - запрещаем часть бомб
            if len(all_planned_bombs) >= 2:
                # Группируем юнитов по близости
                close_groups = []
                for bomber_id, bomber_pos in bomber_positions.items():
                    # Находим бомбу этого юнита
                    bomb_for_bomber = None
                    for cmd in commands:
                        if cmd["id"] == bomber_id and cmd.get("bombs"):
                            bomb_for_bomber = Position(cmd["bombs"][0][0], cmd["bombs"][0][1])
                            break
                    
                    if bomb_for_bomber:
                        # Проверяем, есть ли группа рядом
                        found_group = False
                        for group in close_groups:
                            # Если хотя бы один юнит из группы близок к этому
                            for other_bomber_id, other_bomber_pos in group.items():
                                if bomber_pos.manhattan_distance(other_bomber_pos) <= 2:
                                    group[bomber_id] = bomber_pos
                                    found_group = True
                                    break
                            if found_group:
                                break
                        
                        if not found_group:
                            close_groups.append({bomber_id: bomber_pos})
            
            # Проверяем каждую команду с бомбой на безопасность относительно всех других планируемых бомб
            safe_commands = []
            for cmd in commands:
                if cmd.get("bombs"):
                    bomb_pos = Position(cmd["bombs"][0][0], cmd["bombs"][0][1])
                    bomber_id = cmd["id"]
                    
                    # Получаем список других планируемых бомб (исключая текущую)
                    other_planned_bombs = [b for b in all_planned_bombs if b != bomb_pos]
                    
                    # Получаем планируемые пути других юнитов (исключая текущего)
                    other_planned_paths = {bid: path for bid, path in planned_paths.items() if bid != bomber_id}
                    
                    # Проверяем безопасность с учетом всех других планируемых бомб и путей
                    if not self.will_bomb_hit_other_bombers(bomb_pos, bomber_id, planned_bombs=other_planned_bombs, planned_paths=other_planned_paths):
                        safe_commands.append(cmd)
                    else:
                        print(f"[BOMB SAFETY] Removing unsafe bomb command for {bomber_id} at {bomb_pos}")
                        # Удаляем бомбу из команды, но оставляем путь если есть
                        safe_cmd = {"id": cmd["id"], "path": cmd.get("path", []), "bombs": []}
                        safe_commands.append(safe_cmd)
                else:
                    safe_commands.append(cmd)
            
            commands = safe_commands
        
        if commands:
            print(f"[MOVE] Sending {len(commands)} commands")
            for cmd in commands:
                if cmd.get("bombs"):
                    print(f"  [BOMB] {cmd['id']} placing bomb at {cmd['bombs'][0]}")
            return self.move_bombers(commands)
        return {}
    
    def update_booster_stats(self):
        """Обновить статистику улучшений из состояния игры"""
        try:
            data = self.get_boosters()
            if not data:
                return  # Нет данных - возможно игра еще не началась
            state = data.get("state", {})
            if not state:
                return

            # Отслеживаем изменения очков
            current_points = state.get("points", 0)
            points_diff = current_points - self.last_points

            if points_diff > 0:
                print(f"[SCORE] Points gained: +{points_diff} (total: {current_points})")
            elif points_diff < 0:
                print(f"[SCORE] Points spent: {points_diff} (total: {current_points})")

            self.last_points = current_points

            self.bomb_range = state.get("bomb_range", 1)
            self.bomb_delay = state.get("bomb_delay", 8000)  # В миллисекундах
            self.bomb_delay_level = (8000 - self.bomb_delay) // 2000  # Уровень улучшения
            speed = state.get("speed", 2)
            self.speed_level = max(0, speed - 2)  # Базовая скорость 2

            # Акробатика
            self.acrobatics_level = 0
            if state.get("can_pass_bombs", False):
                self.acrobatics_level = 1
            if state.get("can_pass_obstacles", False):
                self.acrobatics_level = 2
            if state.get("can_pass_walls", False):
                self.acrobatics_level = 3

        except Exception as e:
            print(f"[WARN] Error updating booster stats: {e}")
            pass  # Игнорируем ошибки
    
    def should_buy_booster(self) -> Optional[int]:
        """Логика покупки бустеров"""
        if time.time() < self.booster_check_skip_until: return None
        
        try:
            data = self.get_boosters()
            if not data: 
                # Если нет данных, пропускаем проверку на 10 секунд
                self.booster_check_skip_until = time.time() + 10
                return None
                
            state = data.get("state", {})
            if not state:
                return None
                
            points = state.get("points", 0)
            if points < 1: return None
            
            # Обновляем статы
            self.update_booster_stats()
            
            available = data.get("available", [])
            if not available:
                return None
            
            # Приоритеты: Range > Speed > Bombs > Acrobatics
            # Используем названия типов из API ответа
            for b in available:
                if b.get('type') == 'bomb_range' and b.get('cost', 999) <= points:
                    return BoosterType.BOMB_RANGE
            if self.speed_level < 3:
                for b in available:
                    if b.get('type') == 'speed' and b.get('cost', 999) <= points:
                        return BoosterType.SPEED
            for b in available:
                if b.get('type') == 'bomb_count' and b.get('cost', 999) <= points:
                    return BoosterType.POCKETS
            if self.acrobatics_level < 1 and points >= 2:
                for b in available:
                    if b.get('type') == 'passability' and b.get('cost', 999) <= points:
                        return BoosterType.ACROBATICS
                
            return None
            
        except Exception:
            # При любой ошибке пропускаем проверку на 10 секунд
            self.booster_check_skip_until = time.time() + 10
            return None

    def run(self, verbose: bool = False):
        """Главный цикл"""
        print("[CLIENT] Starting client run()...")
        iteration = 0
        last_booster_check = 0
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

                # Покупка бустеров раз в 5 сек
                result = {}
                if current_time - last_booster_check > 5.0:
                    last_booster_check = current_time
                    booster = self.should_buy_booster()
                    if booster:
                        self.buy_booster(booster)
                        if verbose: print(f"Куплен бустер: {booster}")

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
