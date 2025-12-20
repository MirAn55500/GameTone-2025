"""
Модуль для поиска пути и навигации
"""

import heapq
import time
from typing import List, Optional, Set, Tuple
from models import Position, Bomber, Bomb, Mob, Enemy


class PathFinder:
    """Класс для поиска пути с учетом опасностей"""
    
    def __init__(self, map_size: Tuple[int, int], walls: Set[Position], 
                 obstacles: Set[Position], bombs: List[Bomb], mobs: List[Mob],
                 enemies: List[Enemy], bomb_range: int, bomb_delay: int,
                 speed_level: int, acrobatics_level: int,
                 safe_pathfinder=None, use_advanced_logic: bool = False):
        self.map_size = map_size
        self.walls = walls
        self.obstacles = obstacles
        self.bombs = bombs
        self.mobs = mobs
        self.enemies = enemies
        self.bomb_range = bomb_range
        self.bomb_delay = bomb_delay
        self.speed_level = speed_level
        self.acrobatics_level = acrobatics_level
        self.safe_pathfinder = safe_pathfinder
        self.use_advanced_logic = use_advanced_logic
    
    def is_valid_position(self, pos: Position) -> bool:
        """Проверить, находится ли позиция в пределах карты и не является ли стеной"""
        if not self.map_size:
            return False
        if not (0 <= pos.x < self.map_size[0] and 0 <= pos.y < self.map_size[1]):
            return False
        return pos not in self.walls
    
    def get_blast_danger(self, pos: Position, time_offset: float = 0.0) -> float:
        """Получить время до взрыва в этой позиции (inf если безопасно)"""
        min_timer = float('inf')
        
        for bomb in self.bombs:
            # Проверяем, попадает ли позиция в радиус взрыва
            if bomb.pos.x == pos.x:
                y_dist = abs(bomb.pos.y - pos.y)
                if y_dist <= bomb.range:
                    # Проверяем, нет ли стены между бомбой и позицией
                    blocked = False
                    step = 1 if pos.y > bomb.pos.y else -1
                    for y in range(bomb.pos.y + step, pos.y, step):
                        if Position(bomb.pos.x, y) in self.walls:
                            blocked = True
                            break
                    if not blocked:
                        timer_ms = bomb.timer * 1000 if bomb.timer < 100 else bomb.timer
                        timer_sec = timer_ms / 1000.0 - time_offset
                        min_timer = min(min_timer, timer_sec)
            elif bomb.pos.y == pos.y:
                x_dist = abs(bomb.pos.x - pos.x)
                if x_dist <= bomb.range:
                    blocked = False
                    step = 1 if pos.x > bomb.pos.x else -1
                    for x in range(bomb.pos.x + step, pos.x, step):
                        if Position(x, bomb.pos.y) in self.walls:
                            blocked = True
                            break
                    if not blocked:
                        timer_ms = bomb.timer * 1000 if bomb.timer < 100 else bomb.timer
                        timer_sec = timer_ms / 1000.0 - time_offset
                        min_timer = min(min_timer, timer_sec)
        
        return min_timer
    
    def find_path(self, start: Position, target: Position, max_steps: int = 20) -> Optional[List[Position]]:
        """
        A* поиск пути с учетом опасности от ВСЕХ бомб.
        Использует продвинутую логику Space-Time A*, если доступна.
        """
        if start == target:
            return [target]
        
        # Используем продвинутую логику, если доступна
        if self.use_advanced_logic and self.safe_pathfinder:
            path = self.safe_pathfinder.find_path(
                (start.x, start.y), (target.x, target.y), max_ticks=max_steps * 2
            )
            if path:
                return [Position(x, y) for x, y in path]
            return None
        
        # Heuristic
        def h(p):
            return abs(p.x - target.x) + abs(p.y - target.y)
        
        open_set = [(h(start), 0, start, [start])]  # f_score, g_score, current, path
        visited = {start: 0}  # pos -> g_score
        
        start_time = time.time()
        
        while open_set:
            if time.time() - start_time > 0.1:  # Timeout
                break
            
            f, g, current, path = heapq.heappop(open_set)
            
            if len(path) > max_steps:
                continue
            
            if current == target:
                return path
            
            # Neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                
                if not self.is_valid_position(neighbor):
                    continue
                
                # Check danger from ALL bombs
                # Предполагаем скорость 2-3 клетки/сек. Время прибытия ~ g * 0.3
                arrival_time = (g + 1) * 0.3
                danger_timer = self.get_blast_danger(neighbor, time_offset=0)
                
                cost = 1
                
                if danger_timer < float('inf'):
                    # КРИТИЧНО: избегаем позиций, где бомба взорвется пока мы там
                    if danger_timer < arrival_time + 2.0:  # Увеличили запас безопасности
                        if danger_timer > arrival_time - 0.5:  # Бомба взорвется пока мы там или сразу после
                            cost += 10000  # ОЧЕНЬ ОПАСНО, строго избегаем
                        elif danger_timer < arrival_time - 0.5:
                            # Взрывается до нас, но может быть опасно
                            cost += 100  # Средний штраф
                
                # КРИТИЧНО: Проверяем опасность от ВСЕХ бомб на карте
                neighbor_danger = self.get_blast_danger(neighbor, time_offset=arrival_time)
                if neighbor_danger < float('inf'):
                    if neighbor_danger < arrival_time + 1.0:
                        cost += 10000
                    elif neighbor_danger < arrival_time + 2.0:
                        cost += 1000
                
                # Избегаем мобов (особенно призраков!)
                for mob in self.mobs:
                    if mob.safe_time > 0:
                        continue
                    
                    dist_to_mob = mob.pos.manhattan_distance(neighbor)
                    
                    if mob.pos == neighbor:
                        cost += 1000
                    elif mob.type == "ghost":
                        if dist_to_mob <= 3:
                            cost += 200
                        elif dist_to_mob <= 6:
                            cost += 50
                        elif dist_to_mob <= 10:
                            cost += 10
                    else:
                        if dist_to_mob <= 2:
                            cost += 20
                        elif dist_to_mob <= 3:
                            cost += 5
                
                # Избегаем клеток с бомбами (нельзя наступить, если нет акробатики)
                bomb_at_pos = any(b.pos == neighbor for b in self.bombs)
                if bomb_at_pos and self.acrobatics_level < 1:
                    continue
                
                new_g = g + cost
                
                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (new_g + h(neighbor), new_g, neighbor, new_path))
        
        return None
    
    def find_escape_path_after_bomb(self, bomber: Bomber, bomb_pos: Position) -> Optional[List[Position]]:
        """Найти путь отхода после установки бомбы"""
        bomb_timer = self.bomb_delay / 1000.0
        speed = 2 + self.speed_level
        
        # Ищем безопасные позиции вне радиуса взрыва
        safe_positions = []
        for dx in range(-self.bomb_range - 2, self.bomb_range + 3):
            for dy in range(-self.bomb_range - 2, self.bomb_range + 3):
                if abs(dx) + abs(dy) > self.bomb_range + 2:
                    escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                    if not self.is_valid_position(escape_pos):
                        continue
                    
                    # Проверяем, что позиция не в радиусе взрыва
                    in_blast = False
                    if escape_pos.x == bomb_pos.x and abs(escape_pos.y - bomb_pos.y) <= self.bomb_range:
                        in_blast = True
                    elif escape_pos.y == bomb_pos.y and abs(escape_pos.x - bomb_pos.x) <= self.bomb_range:
                        in_blast = True
                    
                    if not in_blast:
                        # Проверяем безопасность от других бомб
                        danger = self.get_blast_danger(escape_pos)
                        if danger > bomb_timer + 0.5:
                            escape_path = self.find_path(bomber.pos, escape_pos, max_steps=10)
                            if escape_path and len(escape_path) > 1:
                                time_to_escape = len(escape_path) / speed
                                if time_to_escape < bomb_timer - 0.5:
                                    safe_positions.append((escape_path, danger, len(escape_path)))
        
        if safe_positions:
            safe_positions.sort(key=lambda x: x[2])
            return safe_positions[0][0]
        
        return None
    
    def find_safe_escape_from_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[List[Position]]:
        """Найти безопасный путь отступления от призрака"""
        dx_dir = bomber.pos.x - ghost.pos.x
        dy_dir = bomber.pos.y - ghost.pos.y
        
        if dx_dir != 0:
            dx_dir = 1 if dx_dir > 0 else -1
        if dy_dir != 0:
            dy_dir = 1 if dy_dir > 0 else -1
        
        best_escape = None
        max_distance = -1
        
        for distance in range(8, 16):
            for mult_x in [-1, 0, 1]:
                for mult_y in [-1, 0, 1]:
                    if mult_x == 0 and mult_y == 0:
                        continue
                    
                    escape_pos = Position(
                        bomber.pos.x + (dx_dir * distance * mult_x if dx_dir != 0 else distance * mult_x),
                        bomber.pos.y + (dy_dir * distance * mult_y if dy_dir != 0 else distance * mult_y)
                    )
                    
                    if not self.is_valid_position(escape_pos):
                        continue
                    
                    dist_to_ghost = escape_pos.manhattan_distance(ghost.pos)
                    
                    if self.get_blast_danger(escape_pos) < 2.0:
                        continue
                    
                    escape_path = self.find_path(bomber.pos, escape_pos, max_steps=15)
                    if not escape_path or len(escape_path) < 2:
                        continue
                    
                    if dist_to_ghost > max_distance:
                        max_distance = dist_to_ghost
                        best_escape = escape_path
                        
                        if dist_to_ghost >= 12:
                            return best_escape
        
        return best_escape
    
    def check_exit_within_radius(self, pos: Position, radius: int = 5) -> bool:
        """Проверить, есть ли выход (свободная клетка без стен) в радиусе"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                check_pos = Position(pos.x + dx, pos.y + dy)
                if not self.is_valid_position(check_pos):
                    continue
                if check_pos not in self.walls and (check_pos not in self.obstacles or self.acrobatics_level >= 2):
                    walls_around = 0
                    for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = Position(check_pos.x + ndx, check_pos.y + ndy)
                        if neighbor in self.walls:
                            walls_around += 1
                    if walls_around < 3:
                        return True
        return False
