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
    
    def _rate_limit(self):
        """Ограничение скорости запросов"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        if len(self.request_times) >= 4:  # Лимит чуть выше для надежности
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
            return self.api.move_bombers(commands)
        
        for attempt in range(retry_count):
            try:
                self._rate_limit()
                payload = {"bombers": commands}
                response = self.session.post(f"{self.base_url}/api/move", json=payload)
                if response.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(0.5)

    def get_boosters(self, retry_count: int = 2) -> Dict:
        """Получить доступные улучшения"""
        if self.use_local_api and self.api:
            return self.api.get_boosters()

        for attempt in range(retry_count):
            try:
                self._rate_limit()
                response = self.session.get(f"{self.base_url}/api/booster")
                if response.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == retry_count - 1: raise
                time.sleep(0.5)
    
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
        for mob_data in data.get("mobs", []):
            pos = Position(mob_data["pos"][0], mob_data["pos"][1])
            self.mobs.append(Mob(
                mob_data["id"],
                pos,
                mob_data["type"],
                mob_data.get("safe_time", 0)
            ))

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
        """
        min_timer = float('inf')
        is_in_danger = False
        
        for bomb in self.bombs:
            # Если бомба взорвется слишком поздно - игнорируем пока
            if bomb.timer > 10.0: continue
            
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
        """A* поиск пути с учетом опасности"""
        if start == target: return [target]
        
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
                
                # Check danger
                # Предполагаем скорость 2-3 клетки/сек. Время прибытия ~ g * 0.3
                arrival_time = (g + 1) * 0.3 
                danger_timer = self.get_blast_danger(neighbor, time_offset=0) # Check absolute danger time
                
                # Если бомба взрывается в момент прибытия или пока мы там
                # Опасно если: timer < arrival_time + stay_time (скажем 0.5с)
                # И timer > arrival_time (еще не взорвалась до прихода)
                
                cost = 1
                
                if danger_timer < float('inf'):
                    # Если взрывается очень скоро (пока мы идем или стоим)
                    if danger_timer < arrival_time + 1.0: 
                         # Если она взорвется ДО нашего прихода - это ок (уже чисто), НО
                         # взрыв длится какое-то время? Считаем мгновенным.
                         # Но лучше не рисковать ходить в зоны, где таймер < 1-2 сек
                         if danger_timer > arrival_time:
                             cost += 1000 # ОЧЕНЬ ОПАСНО, избегаем
                         else:
                             # Взрывается до нас. Ок, но может быть остаточный эффект?
                             # Считаем безопасно, если таймер < arrival_time - 0.5
                             pass
                
                # Избегаем мобов
                for mob in self.mobs:
                    if mob.pos == neighbor:
                        cost += 500 # Контакт с мобом
                    elif mob.pos.manhattan_distance(neighbor) <= 2:
                        cost += 5 # Рядом с мобом опасно
                
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

    def get_best_target(self, bomber: Bomber) -> Optional[Position]:
        """Выбор лучшей цели для атаки"""
        possible_targets = []
        
        # 1. Используем анализ карты для поиска кластеров
        if self.use_local_api and self.api:
            analysis = self.api.get_map_analysis(force_update=False)
            if analysis and analysis.high_value_targets:
                for pos_tuple, val in analysis.high_value_targets[:5]:
                    pos = Position(pos_tuple[0], pos_tuple[1])
                    if bomber.pos.distance(pos) < 20: # Не слишком далеко
                        possible_targets.append((pos, val * 10)) # Приоритет от кол-ва коробок
        
        # 2. Ближайшие препятствия
        if not possible_targets and self.obstacles:
             # Берем случайные 10 препятствий чтобы не перебирать все
            sample_obstacles = random.sample(list(self.obstacles), min(10, len(self.obstacles)))
            for obs in sample_obstacles:
                possible_targets.append((obs, 5))
                
        # 3. Враги
        for enemy in self.enemies:
            possible_targets.append((enemy.pos, 15)) # Убийство важно
            
        if not possible_targets:
            return None
            
        # Выбор лучшей цели с учетом расстояния
        best_target = None
        best_score = -float('inf')
        
        for target_pos, base_score in possible_targets:
            dist = bomber.pos.manhattan_distance(target_pos)
            if dist == 0: continue
            
            score = base_score - (dist * 0.5)
            
            if score > best_score:
                best_score = score
                best_target = target_pos
                
        return best_target

    def make_move(self) -> Dict:
        """Основная логика хода"""
        commands = []
        
        for bomber in self.bombers:
            if not bomber.alive or not bomber.can_move: continue
            
            my_commands = {"id": bomber.id, "path": [], "bombs": []}
            
            # 1. ПРОВЕРКА БЕЗОПАСНОСТИ (ESCAPE)
            danger_timer = self.get_blast_danger(bomber.pos)
            if danger_timer < 1.5: # Если взрыв через < 1.5 сек
                # СРОЧНО БЕЖАТЬ
                best_escape = None
                max_timer = -1
                
                # Проверяем соседей
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    esc_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if self.is_valid_position(esc_pos):
                        # Может там тоже бомба?
                        t = self.get_blast_danger(esc_pos)
                        if t > max_timer:
                            max_timer = t
                            best_escape = esc_pos
                
                if best_escape and max_timer > danger_timer:
                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [best_escape.x, best_escape.y]]
                    commands.append(my_commands)
                    continue
            
            # 2. АТАКА / ДВИЖЕНИЕ К ЦЕЛИ
            target = self.get_best_target(bomber)
            
            if target:
                # Пытаемся подойти к цели
                # Если цель - препятствие, надо встать РЯДОМ
                target_is_obstacle = target in self.obstacles
                
                move_target = target
                if target_is_obstacle:
                    # Ищем соседнюю клетку
                    min_dist = float('inf')
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        p = Position(target.x + dx, target.y + dy)
                        if self.is_valid_position(p):
                            d = bomber.pos.manhattan_distance(p)
                            if d < min_dist:
                                min_dist = d
                                move_target = p
                
                path = self.find_path(bomber.pos, move_target)
                if path and len(path) > 1:
                    # Ограничиваем длину пути
                    path = path[:10]
                    my_commands["path"] = [[p.x, p.y] for p in path]
                    
                    # Если мы пришли на позицию стрельбы (или близко)
                    if len(path) <= 2 and bomber.bombs_available > 0:
                        # Проверяем, стоит ли ставить бомбу
                        # Если рядом есть препятствия или враги
                        nearby_targets = 0
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                             check = Position(path[-1].x + dx, path[-1].y + dy) # Проверяем вокруг точки назначения
                             if check in self.obstacles: nearby_targets += 1
                        
                        if nearby_targets > 0:
                             # Ставим бомбу в конце пути
                             # ВАЖНО: проверить, сможем ли убежать!
                             # Симуляция: если поставим бомбу тут, будет ли escape?
                             my_commands["bombs"] = [[path[-1].x, path[-1].y]]
                    
                    commands.append(my_commands)
                    continue
            
            # 3. RANDOM ROAM (чтобы не стоять)
            if not my_commands["path"]:
                valid_moves = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    p = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if self.is_valid_position(p) and self.get_blast_danger(p) > 2.0:
                         valid_moves.append(p)
                
                if valid_moves:
                    chosen = random.choice(valid_moves)
                    my_commands["path"] = [[bomber.pos.x, bomber.pos.y], [chosen.x, chosen.y]]
                    commands.append(my_commands)

        return self.move_bombers(commands)

    def should_buy_booster(self) -> Optional[int]:
        """Логика покупки бустеров"""
        if time.time() < self.booster_check_skip_until: return None
        
        try:
            data = self.get_boosters()
            if not data: return None # Handle empty response/error
            state = data.get("state", {})
            points = state.get("points", 0)
            
            if points < 1: return None
            
            # Обновляем статы
            self.bomb_range = state.get("bomb_range", 1)
            self.speed_level = state.get("speed", 2) - 2
            
            available = data.get("available", [])
            
            # Приоритеты: Range > Speed > Bombs
            for b in available:
                if b['type'] == 'bomb_range' and b['cost'] <= points: return BoosterType.BOMB_RANGE
            if self.speed_level < 3:
                for b in available:
                    if b['type'] == 'speed' and b['cost'] <= points: return BoosterType.SPEED
            for b in available:
                if b['type'] == 'pockets' and b['cost'] <= points: return BoosterType.POCKETS
                
            return None
            
        except Exception:
            self.booster_check_skip_until = time.time() + 10
            return None

    def run(self, verbose: bool = False):
        """Главный цикл"""
        print("Запуск клиента...")
        iteration = 0
        last_booster_check = 0
        
        while True:
            try:
                iteration += 1
                self.update_state()
                
                # Покупка бустеров раз в 5 сек
                if time.time() - last_booster_check > 5:
                    last_booster_check = time.time()
                    booster = self.should_buy_booster()
                    if booster: 
                        self.buy_booster(booster)
                        if verbose: print(f"Куплен бустер: {booster}")

                result = self.make_move()
                if verbose and iteration % 10 == 0:
                    print(f"Iter {iteration}: Active bombers {len([b for b in self.bombers if b.alive])}")
                    if result.get('errors'): print("Errors:", result['errors'])

                time.sleep(0.5) # Не частить
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://games.datsteam.dev")
    
    use_local_api = False
    if "--local" in sys.argv: use_local_api = True
    
    if not api_key:
        print("API_KEY required")
        sys.exit(1)
        
    client = GameClient(api_key, base_url, use_local_api=use_local_api)
    client.run(verbose=True)
