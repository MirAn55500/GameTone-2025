"""
Геймтон DatsJingleBang - Клиент для игры
Стратегия: максимизация очков через эффективное уничтожение препятствий
"""

import requests
import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


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
    
    def __init__(self, api_key: str, base_url: str = "https://games.datsteam.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})
        
        # Состояние игры
        self.map_size: Optional[Tuple[int, int]] = None
        self.walls: List[Position] = []
        self.obstacles: List[Position] = []
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
        
        # Время последнего запроса (для ограничения 3 запроса/сек)
        self.last_request_time = 0
        self.request_times = []
    
    def _rate_limit(self):
        """Ограничение скорости запросов (3 в секунду)"""
        now = time.time()
        # Удаляем запросы старше 1 секунды
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        if len(self.request_times) >= 3:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_times.append(time.time())
    
    def get_arena(self) -> Dict:
        """Получить состояние арены"""
        self._rate_limit()
        response = self.session.get(f"{self.base_url}/api/arena")
        response.raise_for_status()
        return response.json()
    
    def move_bombers(self, commands: List[Dict]) -> Dict:
        """Отправить команды движения"""
        self._rate_limit()
        payload = {"bombers": commands}
        response = self.session.post(f"{self.base_url}/api/move", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_boosters(self) -> Dict:
        """Получить доступные улучшения"""
        self._rate_limit()
        response = self.session.get(f"{self.base_url}/api/booster")
        response.raise_for_status()
        return response.json()
    
    def buy_booster(self, booster_type: int) -> Dict:
        """Купить улучшение"""
        self._rate_limit()
        payload = {"booster": booster_type}
        response = self.session.post(f"{self.base_url}/api/booster", json=payload)
        response.raise_for_status()
        return response.json()
    
    def update_state(self):
        """Обновить состояние игры"""
        data = self.get_arena()
        
        # Размер карты
        if "map_size" in data:
            self.map_size = tuple(data["map_size"])
        
        # Стены
        self.walls = [Position(x, y) for x, y in data.get("arena", {}).get("walls", [])]
        
        # Препятствия
        self.obstacles = [Position(x, y) for x, y in data.get("arena", {}).get("obstacles", [])]
        
        # Бомбы
        self.bombs = []
        for bomb_data in data.get("arena", {}).get("bombs", []):
            pos = Position(bomb_data["pos"][0], bomb_data["pos"][1])
            self.bombs.append(Bomb(pos, bomb_data["timer"], bomb_data["range"]))
        
        # Юниты
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
        
        # Враги
        self.enemies = []
        for enemy_data in data.get("enemies", []):
            pos = Position(enemy_data["pos"][0], enemy_data["pos"][1])
            self.enemies.append(Enemy(
                enemy_data["id"],
                pos,
                enemy_data.get("safe_time", 0)
            ))
        
        # Мобы
        self.mobs = []
        for mob_data in data.get("mobs", []):
            pos = Position(mob_data["pos"][0], mob_data["pos"][1])
            self.mobs.append(Mob(
                mob_data["id"],
                pos,
                mob_data["type"],
                mob_data.get("safe_time", 0)
            ))
    
    def is_valid_position(self, pos: Position) -> bool:
        """Проверить, валидна ли позиция"""
        if not self.map_size:
            return False
        
        if pos.x < 0 or pos.y < 0 or pos.x >= self.map_size[0] or pos.y >= self.map_size[1]:
            return False
        
        # Проверка стен
        if pos in self.walls:
            return False
        
        # Проверка препятствий (если нет акробатики уровня 2+)
        if self.acrobatics_level < 2 and pos in self.obstacles:
            return False
        
        return True
    
    def is_safe_position(self, pos: Position, time_ahead: float = 0) -> bool:
        """Проверить, безопасна ли позиция (нет бомб, которые взорвутся)"""
        for bomb in self.bombs:
            # Время до взрыва
            explosion_time = bomb.timer - time_ahead
            
            # Если бомба уже взорвалась или взорвется слишком поздно
            if explosion_time <= 0 or explosion_time > 10:
                continue
            
            # Проверка попадания в радиус взрыва (взрыв в виде креста)
            # Горизонтальный луч
            if pos.y == bomb.pos.y:
                if abs(pos.x - bomb.pos.x) <= bomb.range:
                    return False
            # Вертикальный луч
            if pos.x == bomb.pos.x:
                if abs(pos.y - bomb.pos.y) <= bomb.range:
                    return False
        
        # Проверка на мобов (опасны при контакте)
        for mob in self.mobs:
            if mob.safe_time == 0 and mob.pos == pos:
                return False
        
        return True
    
    def estimate_path_time(self, path: List[Position], speed: int = 2) -> float:
        """Оценить время прохождения пути в секундах"""
        if not path:
            return 0
        # Скорость в клетках в секунду
        return len(path) / speed
    
    def find_path(self, start: Position, target: Position, max_steps: int = 30) -> Optional[List[Position]]:
        """Найти путь от start до target (A* алгоритм)"""
        if start == target:
            return [target]
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: start.manhattan_distance(target)}
        
        while open_set:
            current = min(open_set, key=lambda p: f_score.get(p, float('inf')))
            
            if current == target:
                # Восстановить путь
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))[:max_steps]
            
            open_set.remove(current)
            
            # Проверить соседние клетки
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                
                if not self.is_valid_position(neighbor):
                    continue
                
                # Проверка бомб (если нет акробатики уровня 1+)
                if self.acrobatics_level < 1:
                    if any(bomb.pos == neighbor for bomb in self.bombs):
                        continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + neighbor.manhattan_distance(target)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        return None
    
    def count_obstacles_in_range(self, bomb_pos: Position, bomb_range: int = None) -> int:
        """Подсчитать количество препятствий в радиусе взрыва бомбы"""
        if bomb_range is None:
            bomb_range = self.bomb_range
        
        count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            for r in range(1, bomb_range + 1):
                check_pos = Position(bomb_pos.x + dx * r, bomb_pos.y + dy * r)
                if check_pos in self.obstacles:
                    count += 1
                    break  # Луч взрыва останавливается на препятствии
                elif check_pos in self.walls:
                    break  # Луч останавливается на стене
        
        return count
    
    def find_mob_target(self, bomber: Bomber) -> Optional[Position]:
        """Найти ближайшего моба для атаки (10 очков за моба)"""
        if not bomber.alive or not bomber.can_move:
            return None
        
        # Ищем ближайшего моба, который не спит
        active_mobs = [mob for mob in self.mobs if mob.safe_time == 0]
        if not active_mobs:
            return None
        
        nearest_mob = min(active_mobs, key=lambda m: bomber.pos.manhattan_distance(m.pos))
        
        # Проверяем, можем ли мы безопасно подойти
        # Мобы опасны при контакте, но можно взорвать бомбой
        # Ищем позицию рядом с мобом для размещения бомбы
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(nearest_mob.pos.x + dx, nearest_mob.pos.y + dy)
            if self.is_valid_position(bomb_pos) and self.is_safe_position(bomb_pos, time_ahead=8.0):
                path = self.find_path(bomber.pos, bomb_pos, max_steps=30)
                if path:
                    return bomb_pos
        
        return None
    
    def find_enemy_target(self, bomber: Bomber) -> Optional[Position]:
        """Найти ближайшего врага для атаки (10 очков за убийство)"""
        if not bomber.alive or not bomber.can_move:
            return None
        
        if not self.enemies:
            return None
        
        # Ищем ближайшего врага
        nearest_enemy = min(self.enemies, key=lambda e: bomber.pos.manhattan_distance(e.pos))
        
        # Пытаемся предсказать, где будет враг, и поставить бомбу
        # Для простоты ставим бомбу рядом с текущей позицией врага
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(nearest_enemy.pos.x + dx, nearest_enemy.pos.y + dy)
            if self.is_valid_position(bomb_pos) and self.is_safe_position(bomb_pos, time_ahead=8.0):
                path = self.find_path(bomber.pos, bomb_pos, max_steps=30)
                if path and bomber.pos.manhattan_distance(bomb_pos) <= 15:  # Не слишком далеко
                    return bomb_pos
        
        return None
    
    def find_best_obstacle_target(self, bomber: Bomber) -> Optional[Tuple[Position, Position]]:
        """Найти лучшую цель для уничтожения препятствий
        Возвращает (позиция для бомбы, позиция цели)"""
        if not bomber.alive or not bomber.can_move:
            return None
        
        best_bomb_pos = None
        best_target = None
        best_score = 0
        bomb_range = 1  # Базовый радиус, можно получить из состояния
        
        # Ищем позиции для размещения бомб рядом с препятствиями
        checked_bomb_positions = set()
        
        for obstacle in self.obstacles:
            # Проверяем все позиции рядом с препятствием, где можно поставить бомбу
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                bomb_pos = Position(obstacle.x + dx, obstacle.y + dy)
                
                # Пропускаем если уже проверяли
                if bomb_pos in checked_bomb_positions:
                    continue
                checked_bomb_positions.add(bomb_pos)
                
                # Проверяем валидность позиции для бомбы
                if not self.is_valid_position(bomb_pos):
                    continue
                
                # Проверяем безопасность (нет других бомб, которые взорвутся)
                if not self.is_safe_position(bomb_pos, time_ahead=8.0):
                    continue
                
                # Подсчитываем количество препятствий в радиусе
                destroyed = self.count_obstacles_in_range(bomb_pos, self.bomb_range)
                
                if destroyed == 0:
                    continue
                
                # Очки: 1 + 2 + 3 + 4 = 10 максимум за 4+ препятствия
                score = sum(range(1, min(destroyed + 1, 5)))
                
                # Бонус за близость к юниту
                distance = bomber.pos.manhattan_distance(bomb_pos)
                score = score * (1.0 - distance * 0.01)  # Небольшой бонус за близость
                
                if score > best_score:
                    best_score = score
                    best_bomb_pos = bomb_pos
                    best_target = obstacle
        
        if best_bomb_pos and best_target:
            return (best_bomb_pos, best_target)
        return None
    
    def plan_bomb_placement(self, bomber: Bomber, bomb_pos: Position, target: Position) -> Tuple[List[Position], List[Position]]:
        """Спланировать путь и размещение бомб"""
        if not bomb_pos:
            return [], []
        
        # Находим путь к позиции бомбы
        path = self.find_path(bomber.pos, bomb_pos, max_steps=30)
        if not path:
            return [], []
        
        # Размещаем бомбу на целевой позиции
        bomb_positions = [bomb_pos] if bomber.bombs_available > 0 else []
        
        return path, bomb_positions
    
    def should_buy_booster(self) -> Optional[int]:
        """Определить, какое улучшение купить"""
        try:
            boosters_data = self.get_boosters()
            available = boosters_data.get("available", [])
            state = boosters_data.get("state", {})
            points = state.get("points", 0)
            
            if points < 1:
                return None
            
            # Обновляем текущие уровни из состояния
            self.speed_level = state.get("speed", 2) - 2  # Базовая скорость 2
            self.bomb_range = state.get("bomb_range", 1)
            self.bomb_delay = state.get("bomb_delay", 8000)
            self.bomb_delay_level = (8000 - self.bomb_delay) // 2000  # Базовый таймер 8 сек
            self.acrobatics_level = 0
            if state.get("can_pass_bombs", False):
                self.acrobatics_level = 1
            if state.get("can_pass_obstacles", False):
                self.acrobatics_level = 2
            if state.get("can_pass_walls", False):
                self.acrobatics_level = 3
            
            # Приоритет улучшений (оптимизированная стратегия):
            # 1. Радиус бомбы - КРИТИЧНО для максимизации очков (больше препятствий за взрыв)
            for booster in available:
                if booster.get("type") == "bomb_range" and booster.get("cost", 0) <= points:
                    return BoosterType.BOMB_RANGE
            
            # 2. Скорость (до 3 уровня) - для быстрого перемещения и избегания опасностей
            if self.speed_level < 3:
                for booster in available:
                    if booster.get("type") == "speed" and booster.get("cost", 0) <= points:
                        return BoosterType.SPEED
            
            # 3. Количество бомб - для большего урона и эффективности
            for booster in available:
                if booster.get("type") == "pockets" and booster.get("cost", 0) <= points:
                    return BoosterType.POCKETS
            
            # 4. Акробатика (уровень 1) - для прохода через бомбы (важно для безопасности)
            if self.acrobatics_level < 1 and points >= 2:
                for booster in available:
                    if booster.get("type") == "acrobatics" and booster.get("cost", 0) <= points:
                        return BoosterType.ACROBATICS
            
            # 5. Зрение - для лучшей видимости и планирования
            for booster in available:
                if booster.get("type") == "vision" and booster.get("cost", 0) <= points:
                    return BoosterType.VISION
            
            # 6. Фитиль бомбы (до 3 уровня) - для быстрых взрывов
            if self.bomb_delay_level < 3:
                for booster in available:
                    if booster.get("type") == "bomb_delay" and booster.get("cost", 0) <= points:
                        return BoosterType.BOMB_DELAY
            
            # 7. Броня - для защиты от случайных взрывов
            for booster in available:
                if booster.get("type") == "armor" and booster.get("cost", 0) <= points:
                    return BoosterType.ARMOR
            
            return None
        except Exception as e:
            print(f"Ошибка при проверке улучшений: {e}")
            return None
    
    def find_safe_escape_path(self, bomber: Bomber) -> Optional[List[Position]]:
        """Найти безопасный путь для отступления от бомб"""
        # Получаем текущую скорость
        speed = 2 + self.speed_level
        
        # Ищем позиции, которые будут безопасны
        safe_positions = []
        
        # Ищем в радиусе обзора (обычно 5, но может быть больше с улучшениями)
        search_radius = 10
        
        for x in range(max(0, bomber.pos.x - search_radius), min(self.map_size[0], bomber.pos.x + search_radius + 1)):
            for y in range(max(0, bomber.pos.y - search_radius), min(self.map_size[1], bomber.pos.y + search_radius + 1)):
                pos = Position(x, y)
                if not self.is_valid_position(pos):
                    continue
                
                # Проверяем путь к этой позиции
                path = self.find_path(bomber.pos, pos, max_steps=30)
                if not path:
                    continue
                
                # Оцениваем время до достижения позиции
                path_time = self.estimate_path_time(path, speed)
                
                # Проверяем безопасность с учетом времени движения
                if self.is_safe_position(pos, time_ahead=path_time + 8.0):
                    safe_positions.append((pos, len(path)))
        
        if not safe_positions:
            return None
        
        # Выбираем ближайшую безопасную позицию
        nearest_safe, _ = min(safe_positions, key=lambda item: item[1])
        return self.find_path(bomber.pos, nearest_safe, max_steps=30)
    
    def make_move(self) -> Dict:
        """Принять решение о движении"""
        commands = []
        
        for bomber in self.bombers:
            if not bomber.alive or not bomber.can_move:
                continue
            
            # Проверяем безопасность текущей позиции
            if not self.is_safe_position(bomber.pos, time_ahead=8.0):
                # Нужно убегать!
                escape_path = self.find_safe_escape_path(bomber)
                if escape_path:
                    command = {
                        "id": bomber.id,
                        "path": [[p.x, p.y] for p in escape_path],
                        "bombs": []
                    }
                    commands.append(command)
                    continue
            
            # Стратегия приоритетов:
            # 1. Препятствия (основной источник очков, особенно группы)
            target_result = self.find_best_obstacle_target(bomber)
            
            if target_result:
                bomb_pos, target = target_result
                path, bomb_positions = self.plan_bomb_placement(bomber, bomb_pos, target)
                if path:
                    command = {
                        "id": bomber.id,
                        "path": [[p.x, p.y] for p in path],
                        "bombs": [[p.x, p.y] for p in bomb_positions if bomber.bombs_available > 0]
                    }
                    commands.append(command)
                    continue
            
            # 2. Мобы (10 очков, но опасны)
            mob_bomb_pos = self.find_mob_target(bomber)
            if mob_bomb_pos:
                path = self.find_path(bomber.pos, mob_bomb_pos, max_steps=30)
                if path:
                    bombs = [[mob_bomb_pos.x, mob_bomb_pos.y]] if bomber.bombs_available > 0 else []
                    command = {
                        "id": bomber.id,
                        "path": [[p.x, p.y] for p in path],
                        "bombs": bombs
                    }
                    commands.append(command)
                    continue
            
            # 3. Враги (10 очков, но сложнее попасть)
            enemy_bomb_pos = self.find_enemy_target(bomber)
            if enemy_bomb_pos:
                path = self.find_path(bomber.pos, enemy_bomb_pos, max_steps=30)
                if path:
                    bombs = [[enemy_bomb_pos.x, enemy_bomb_pos.y]] if bomber.bombs_available > 0 else []
                    command = {
                        "id": bomber.id,
                        "path": [[p.x, p.y] for p in path],
                        "bombs": bombs
                    }
                    commands.append(command)
                    continue
            
            # 4. Если нет целей, двигаемся к ближайшему препятствию
            if self.obstacles:
                nearest = min(self.obstacles, key=lambda o: bomber.pos.manhattan_distance(o))
                path = self.find_path(bomber.pos, nearest, max_steps=30)
                if path:
                    command = {
                        "id": bomber.id,
                        "path": [[p.x, p.y] for p in path],
                        "bombs": []
                    }
                    commands.append(command)
        
        if commands:
            return self.move_bombers(commands)
        return {}
    
    def run(self, verbose: bool = False):
        """Основной игровой цикл"""
        print("Запуск клиента игры...")
        print("Нажмите Ctrl+C для остановки")
        
        last_booster_check = 0
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Обновляем состояние
                self.update_state()
                
                if verbose and iteration % 20 == 0:
                    alive_count = sum(1 for b in self.bombers if b.alive)
                    obstacles_count = len(self.obstacles)
                    bombs_count = len(self.bombs)
                    print(f"[Итерация {iteration}] Живых юнитов: {alive_count}, "
                          f"Препятствий: {obstacles_count}, Бомб: {bombs_count}")
                
                # Покупаем улучшения (не слишком часто)
                current_time = time.time()
                if current_time - last_booster_check >= 3.0:  # Проверяем каждые 3 секунды
                    last_booster_check = current_time
                    booster_type = self.should_buy_booster()
                    if booster_type is not None:
                        try:
                            self.buy_booster(booster_type)
                            booster_names = {
                                BoosterType.POCKETS: "Карманы",
                                BoosterType.BOMB_RANGE: "Радиус бомбы",
                                BoosterType.SPEED: "Скорость",
                                BoosterType.VISION: "Зрение",
                                BoosterType.UNITS: "Юниты",
                                BoosterType.ARMOR: "Броня",
                                BoosterType.BOMB_DELAY: "Фитиль",
                                BoosterType.ACROBATICS: "Акробатика"
                            }
                            print(f"✓ Куплено улучшение: {booster_names.get(booster_type, booster_type)}")
                        except Exception as e:
                            if verbose:
                                print(f"Ошибка при покупке улучшения: {e}")
                
                # Делаем ход
                result = self.make_move()
                if result.get("errors"):
                    if verbose:
                        print(f"Ошибки движения: {result['errors']}")
                
                # Небольшая задержка перед следующим циклом
                time.sleep(0.05)  # 50мс - частота дискретизации игры
                
            except KeyboardInterrupt:
                print("\nОстановка клиента...")
                break
            except requests.exceptions.RequestException as e:
                print(f"Ошибка сети: {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Неожиданная ошибка: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python game_client.py <API_KEY> [BASE_URL] [--verbose]")
        print("Пример: python game_client.py your_api_key")
        print("Или для тестового сервера: python game_client.py your_api_key https://games-test.datsteam.dev")
        print("Для подробного вывода: python game_client.py your_api_key https://games-test.datsteam.dev --verbose")
        sys.exit(1)
    
    api_key = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "https://games.datsteam.dev"
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    client = GameClient(api_key, base_url)
    client.run(verbose=verbose)

