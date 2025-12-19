"""
Продвинутая логика для игры Bomberman
Включает симуляцию цепных реакций, Space-Time A*, и улучшенный выбор целей
"""

import heapq
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TimeBomb:
    """Бомба с учетом времени взрыва (с цепными реакциями)"""
    x: int
    y: int
    range: int
    explosion_tick: int  # Тик, когда взорвется (с учетом цепных реакций)
    original_timer: float  # Оригинальный таймер в секундах


@dataclass
class DangerInterval:
    """Интервал опасности для клетки"""
    start_tick: int
    end_tick: int


class MapPredictor:
    """Предсказатель карты - создает DeathMap с учетом цепных реакций"""
    
    def __init__(self, map_size: Tuple[int, int], walls: Set[Tuple[int, int]], 
                 obstacles: Set[Tuple[int, int]], bomb_delay_ms: int = 8000):
        self.width, self.height = map_size
        self.walls = walls
        self.obstacles = obstacles
        self.bomb_delay_ms = bomb_delay_ms
        self.tick_duration_ms = 50  # Предполагаем, что тик = 50мс
        self.explosion_duration_ticks = 6  # Взрыв длится ~300мс = 6 тиков
        
        # DeathMap: (x, y) -> list of DangerInterval
        self.death_map: Dict[Tuple[int, int], List[DangerInterval]] = {}
        
    def simulate_chain_reactions(self, bombs: List[Dict]) -> List[TimeBomb]:
        """
        Симулирует цепные реакции бомб.
        Если бомба А взрывается и задевает бомбу Б, то Б взрывается в момент А.
        
        Args:
            bombs: Список бомб в формате [{'x': int, 'y': int, 'timer': float, 'range': int}, ...]
        
        Returns:
            Список TimeBomb с скорректированными временами взрыва
        """
        if not bombs:
            return []
        
        # Конвертируем таймеры в тики
        time_bombs = []
        for b in bombs:
            timer_ms = b.get('timer', self.bomb_delay_ms)
            if isinstance(timer_ms, (int, float)):
                ticks_left = max(1, int(timer_ms / self.tick_duration_ms))
            else:
                ticks_left = int(self.bomb_delay_ms / self.tick_duration_ms)
            
            time_bombs.append(TimeBomb(
                x=b['x'],
                y=b['y'],
                range=b.get('range', 3),
                explosion_tick=ticks_left,
                original_timer=timer_ms
            ))
        
        # Сортируем по времени взрыва
        time_bombs.sort(key=lambda b: b.explosion_tick)
        
        # Словарь для быстрого поиска бомб по позиции
        bomb_grid = {(b.x, b.y): b for b in time_bombs}
        processed = set()
        final_bombs = []
        queue = list(time_bombs)
        
        while queue:
            # Берем бомбу, которая взорвется раньше всех
            queue.sort(key=lambda b: b.explosion_tick)
            current = queue.pop(0)
            
            bx, by = current.x, current.y
            if (bx, by) in processed:
                continue
            
            processed.add((bx, by))
            final_bombs.append(current)
            
            # Симулируем лучи взрыва
            explosion_cells = self._get_explosion_cells(bx, by, current.range)
            
            # Если луч задевает другую бомбу, ее время сокращается
            for (cx, cy) in explosion_cells:
                if (cx, cy) in bomb_grid and (cx, cy) not in processed:
                    target_bomb = bomb_grid[(cx, cy)]
                    if target_bomb.explosion_tick > current.explosion_tick:
                        # Цепная реакция!
                        target_bomb.explosion_tick = current.explosion_tick
                        # Возвращаем в очередь для пересчета порядка
                        if target_bomb not in queue:
                            queue.append(target_bomb)
        
        return final_bombs
    
    def _get_explosion_cells(self, bx: int, by: int, radius: int) -> List[Tuple[int, int]]:
        """Возвращает список клеток, которые заденет взрыв"""
        cells = []
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in dirs:
            for r in range(1, radius + 1):
                nx, ny = bx + dx * r, by + dy * r
                
                # Стены останавливают взрыв
                if (nx, ny) in self.walls:
                    break
                
                cells.append((nx, ny))
                
                # Ящики останавливают взрыв (но уничтожаются)
                if (nx, ny) in self.obstacles:
                    break
                
                # Другие бомбы останавливают взрыв (но детонируют)
                # Проверка на бомбы будет в simulate_chain_reactions
                
        return cells
    
    def build_death_map(self, time_bombs: List[TimeBomb]):
        """
        Создает карту опасностей: (x, y) -> list of DangerInterval
        
        Args:
            time_bombs: Список бомб с учетом цепных реакций
        """
        self.death_map = defaultdict(list)
        
        for bomb in time_bombs:
            explosion_tick = bomb.explosion_tick
            cells = self._get_explosion_cells(bomb.x, bomb.y, bomb.range)
            cells.append((bomb.x, bomb.y))  # Сама бомба тоже опасна
            
            for cx, cy in cells:
                # Опасно с момента взрыва и + время огня
                interval = DangerInterval(
                    start_tick=explosion_tick,
                    end_tick=explosion_tick + self.explosion_duration_ticks
                )
                self.death_map[(cx, cy)].append(interval)
        
        # Объединяем перекрывающиеся интервалы
        for pos in self.death_map:
            intervals = sorted(self.death_map[pos], key=lambda i: i.start_tick)
            merged = []
            for interval in intervals:
                if not merged or interval.start_tick > merged[-1].end_tick:
                    merged.append(interval)
                else:
                    merged[-1].end_tick = max(merged[-1].end_tick, interval.end_tick)
            self.death_map[pos] = merged
    
    def is_safe_at_time(self, x: int, y: int, arrival_tick: int, 
                       safety_margin: int = 2) -> bool:
        """
        Проверяет, безопасно ли находиться в (x,y) в момент arrival_tick.
        
        Args:
            x, y: Позиция
            arrival_tick: Тик, когда юнит придет в эту позицию
            safety_margin: Запас безопасности в тиках
        
        Returns:
            True если безопасно, False если опасно
        """
        # Статические препятствия
        if (x, y) in self.walls or (x, y) in self.obstacles:
            return False
        
        # Проверяем интервалы опасности
        intervals = self.death_map.get((x, y), [])
        for interval in intervals:
            # Если мы придем, а там уже горит или скоро взорвется
            if interval.start_tick - safety_margin <= arrival_tick <= interval.end_tick + safety_margin:
                return False
        
        return True
    
    def get_danger_at_time(self, x: int, y: int, arrival_tick: int) -> Optional[int]:
        """
        Возвращает тик, когда в этой позиции будет опасно (или None если безопасно).
        
        Returns:
            Тик начала опасности или None
        """
        intervals = self.death_map.get((x, y), [])
        for interval in intervals:
            if interval.start_tick <= arrival_tick <= interval.end_tick:
                return interval.start_tick
            elif interval.start_tick > arrival_tick:
                return interval.start_tick  # Опасность в будущем
        return None


class TargetSelector:
    """Выбор оптимальных целей для бомб"""
    
    def __init__(self, obstacles: Set[Tuple[int, int]], bomb_range: int):
        self.obstacles = obstacles
        self.bomb_range = bomb_range
    
    def calculate_bomb_score(self, bomb_x: int, bomb_y: int) -> Tuple[int, int]:
        """
        Рассчитывает очки за размещение бомбы в позиции (bomb_x, bomb_y).
        
        Формула очков: 1 ящик = 1, 2 ящика = 3, 3 = 6, 4 = 10 (серия)
        
        Returns:
            (score, box_count) - очки и количество ящиков
        """
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        boxes_hit = []
        
        for dx, dy in dirs:
            for r in range(1, self.bomb_range + 1):
                nx, ny = bomb_x + dx * r, bomb_y + dy * r
                
                if (nx, ny) in self.obstacles:
                    boxes_hit.append((nx, ny))
                    break  # Ящик останавливает взрыв
        
        box_count = len(boxes_hit)
        
        # Формула очков за серию: 1=1, 2=3, 3=6, 4=10
        if box_count == 0:
            score = 0
        elif box_count == 1:
            score = 1
        elif box_count == 2:
            score = 3
        elif box_count == 3:
            score = 6
        elif box_count >= 4:
            score = 10
        else:
            score = box_count
        
        return score, box_count
    
    def find_best_bomb_spots(self, current_pos: Tuple[int, int], 
                            visible_obstacles: Set[Tuple[int, int]],
                            max_distance: int = 10) -> List[Tuple[Tuple[int, int], int, int]]:
        """
        Находит лучшие позиции для размещения бомб.
        
        Returns:
            Список [(pos, score, box_count), ...] отсортированный по score
        """
        candidates = []
        
        # Перебираем позиции рядом с ящиками
        checked_positions = set()
        
        for obs_x, obs_y in visible_obstacles:
            # Проверяем позиции рядом с ящиком
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                bomb_x, bomb_y = obs_x + dx, obs_y + dy
                
                if (bomb_x, bomb_y) in checked_positions:
                    continue
                if (bomb_x, bomb_y) in self.obstacles:
                    continue
                
                checked_positions.add((bomb_x, bomb_y))
                
                # Рассчитываем очки
                score, box_count = self.calculate_bomb_score(bomb_x, bomb_y)
                
                if score > 0:
                    distance = abs(bomb_x - current_pos[0]) + abs(bomb_y - current_pos[1])
                    if distance <= max_distance:
                        # Приоритет: больше очков, меньше расстояние
                        final_score = score * 100 - distance
                        candidates.append(((bomb_x, bomb_y), final_score, box_count))
        
        # Сортируем по убыванию очков
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


class SafePathfinder:
    """Space-Time A* поиск пути с учетом DeathMap"""
    
    def __init__(self, map_predictor: MapPredictor, walls: Set[Tuple[int, int]],
                 obstacles: Set[Tuple[int, int]], enemies: List[Dict],
                 mobs: List[Dict], speed: int = 2):
        self.predictor = map_predictor
        self.walls = walls
        self.obstacles = obstacles
        self.enemies = enemies
        self.mobs = mobs
        self.speed = speed  # Клеток в секунду
        self.tick_duration_ms = 50
        self.ticks_per_cell = max(1, int(1000 / (self.speed * 1000 / self.tick_duration_ms)))
    
    def find_path(self, start: Tuple[int, int], target: Tuple[int, int],
                  max_ticks: int = 40) -> Optional[List[Tuple[int, int]]]:
        """
        Space-Time A* поиск пути.
        
        Args:
            start: Начальная позиция (x, y)
            target: Целевая позиция (x, y)
            max_ticks: Максимальное количество тиков для поиска
        
        Returns:
            Путь как список позиций или None
        """
        if start == target:
            return [target]
        
        def heuristic(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Node = (x, y, tick)
        start_node = (*start, 0)
        queue = [(heuristic(*start, *target), 0, start_node, [start])]
        visited = {start_node: 0}
        came_from = {}
        
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            f_score, g_ticks, (cx, cy, ct), path = heapq.heappop(queue)
            
            if (cx, cy) == target:
                return path
            
            if ct >= max_ticks:
                continue
            
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                
                # Проверка статических препятствий
                if (nx, ny) in self.walls or (nx, ny) in self.obstacles:
                    continue
                
                # Время прибытия в новую позицию
                nt = ct + self.ticks_per_cell
                
                # Проверка безопасности в БУДУЩЕМ
                if not self.predictor.is_safe_at_time(nx, ny, nt, safety_margin=3):
                    continue
                
                # Проверка врагов и мобов
                if self._is_dangerous_position(nx, ny):
                    continue
                
                state = (nx, ny, nt)
                new_g = g_ticks + 1
                
                if state not in visited or new_g < visited[state]:
                    visited[state] = new_g
                    h = heuristic(nx, ny, *target)
                    new_path = path + [(nx, ny)]
                    heapq.heappush(queue, (new_g + h, new_g, state, new_path))
                    came_from[state] = (cx, cy, ct)
        
        return None
    
    def _is_dangerous_position(self, x: int, y: int) -> bool:
        """Проверяет, опасна ли позиция из-за врагов/мобов"""
        for enemy in self.enemies:
            if enemy.get('x') == x and enemy.get('y') == y:
                return True
        
        for mob in self.mobs:
            if mob.get('x') == x and mob.get('y') == y:
                return True
            # Призраки опасны на расстоянии <= 2
            if mob.get('type') == 'ghost':
                dist = abs(x - mob.get('x', 0)) + abs(y - mob.get('y', 0))
                if dist <= 2:
                    return True
        
        return False
    
    def find_sanctuary(self, start: Tuple[int, int], max_ticks: int = 20) -> Optional[Tuple[int, int]]:
        """
        Ищет ближайшую безопасную позицию (Sanctuary).
        Используется в режиме выживания.
        
        Returns:
            Безопасная позиция или None
        """
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = [(0, start, 0)]  # (distance, pos, tick)
        visited = {start}
        
        while queue:
            dist, (cx, cy), ct = heapq.heappop(queue)
            
            if ct >= max_ticks:
                continue
            
            # Проверяем, безопасна ли эта позиция на длительное время
            if self.predictor.is_safe_at_time(cx, cy, ct, safety_margin=5):
                # Проверяем, что будет безопасно еще несколько тиков
                future_safe = True
                for future_tick in range(ct, ct + 10):
                    if not self.predictor.is_safe_at_time(cx, cy, future_tick, safety_margin=3):
                        future_safe = False
                        break
                
                if future_safe and not self._is_dangerous_position(cx, cy):
                    return (cx, cy)
            
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                
                if (nx, ny) in visited:
                    continue
                if (nx, ny) in self.walls or (nx, ny) in self.obstacles:
                    continue
                
                visited.add((nx, ny))
                new_dist = dist + 1
                new_tick = ct + self.ticks_per_cell
                
                if new_tick < max_ticks:
                    heapq.heappush(queue, (new_dist, (nx, ny), new_tick))
        
        return None
