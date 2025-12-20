"""
Модуль для боевых действий (работа с бомбами, врагами, мобами)
"""

from typing import List, Dict, Tuple, Optional, Set
from models import Position, Bomber, Bomb, Enemy, Mob


class CombatManager:
    """Менеджер боевых действий"""
    
    def __init__(self, game_client):
        """Инициализация с ссылкой на GameClient для доступа к методам"""
        self.client = game_client
    
    def find_nearby_ghost(self, bomber: Bomber, max_distance: int = 12) -> Optional[Mob]:
        """Найти ближайшего призрака (ghost) в радиусе обзора"""
        nearest_ghost = None
        min_dist = float('inf')
        
        for mob in self.client.mobs:
            if mob.safe_time > 0:
                continue
            if mob.type != "ghost":
                continue
            
            dist = bomber.pos.manhattan_distance(mob.pos)
            if dist <= max_distance and dist < min_dist:
                min_dist = dist
                nearest_ghost = mob
        
        return nearest_ghost
    
    def find_nearby_enemy(self, bomber: Bomber, max_distance: int = 3) -> Optional[Enemy]:
        """Найти ближайшего врага или моба"""
        nearest = None
        min_dist = float('inf')
        
        # Проверяем врагов
        for enemy in self.client.enemies:
            dist = bomber.pos.manhattan_distance(enemy.pos)
            if dist <= max_distance and dist < min_dist:
                min_dist = dist
                nearest = enemy
        
        # Проверяем мобов (опасны при контакте)
        for mob in self.client.mobs:
            if mob.safe_time > 0:
                continue
            dist = bomber.pos.manhattan_distance(mob.pos)
            if dist <= max_distance:
                priority_dist = dist
                if mob.type == "ghost":
                    priority_dist = dist - 2  # Бонус приоритета
                
                if priority_dist < min_dist:
                    min_dist = priority_dist
                    nearest = Enemy(mob.id, mob.pos, mob.safe_time)
        
        return nearest
    
    def will_bomb_hit_other_bombers(self, bomb_pos: Position, bomber_id: str, 
                                    planned_bombs: Optional[List[Position]] = None,
                                    planned_paths: Optional[Dict[str, List[Position]]] = None) -> bool:
        """Проверить, попадут ли другие юниты под взрыв бомбы"""
        bomb_timer = self.client.bomb_delay / 1000.0
        speed = 2 + (self.client.booster_manager.speed_level if self.client.booster_manager else 0)
        
        all_bomb_positions = [bomb_pos]
        if planned_bombs:
            all_bomb_positions.extend(planned_bombs)
        
        for other_bomber in self.client.bombers:
            if other_bomber.id == bomber_id or not other_bomber.alive:
                continue
            
            other_pos = other_bomber.pos
            
            if planned_paths and other_bomber.id in planned_paths:
                planned_path = planned_paths[other_bomber.id]
                if planned_path and len(planned_path) > 0:
                    final_path_pos = Position(planned_path[-1][0], planned_path[-1][1])
                    if final_path_pos.manhattan_distance(bomb_pos) > other_pos.manhattan_distance(bomb_pos):
                        other_pos = final_path_pos
            
            in_range_of_any_bomb = False
            dangerous_bomb_pos = None
            
            for check_bomb_pos in all_bomb_positions:
                in_range = False
                if other_pos.y == check_bomb_pos.y and abs(other_pos.x - check_bomb_pos.x) <= self.client.bomb_range:
                    blocked = False
                    step = 1 if other_pos.x > check_bomb_pos.x else -1
                    for x in range(check_bomb_pos.x + step, other_pos.x, step):
                        if Position(x, check_bomb_pos.y) in self.client.walls or Position(x, check_bomb_pos.y) in self.client.obstacles:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                elif other_pos.x == check_bomb_pos.x and abs(other_pos.y - check_bomb_pos.y) <= self.client.bomb_range:
                    blocked = False
                    step = 1 if other_pos.y > check_bomb_pos.y else -1
                    for y in range(check_bomb_pos.y + step, other_pos.y, step):
                        if Position(check_bomb_pos.x, y) in self.client.walls or Position(check_bomb_pos.x, y) in self.client.obstacles:
                            blocked = True
                            break
                    if not blocked:
                        in_range = True
                
                if in_range:
                    in_range_of_any_bomb = True
                    dangerous_bomb_pos = check_bomb_pos
                    break
            
            if in_range_of_any_bomb:
                safe_escape_found = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    escape_pos = Position(other_pos.x + dx, other_pos.y + dy)
                    if not self.client.is_valid_position(escape_pos):
                        continue
                    
                    escape_in_blast = False
                    for check_bomb_pos in all_bomb_positions:
                        if escape_pos.x == check_bomb_pos.x and abs(escape_pos.y - check_bomb_pos.y) <= self.client.bomb_range:
                            escape_in_blast = True
                            break
                        elif escape_pos.y == check_bomb_pos.y and abs(escape_pos.x - check_bomb_pos.x) <= self.client.bomb_range:
                            escape_in_blast = True
                            break
                    
                    if not escape_in_blast:
                        time_to_escape = 1.0 / speed
                        if time_to_escape < bomb_timer - 0.3:
                            safe_escape_found = True
                            break
                
                if not safe_escape_found:
                    return True
        
        return False
    
    def can_safely_place_bomb(self, bomber: Bomber, bomb_pos: Position, aggressive: bool = False) -> Tuple[bool, Optional[List[Position]]]:
        """Проверить, можно ли безопасно поставить бомбу и убежать"""
        if bomber.bombs_available == 0:
            return False, None
        
        if self.will_bomb_hit_other_bombers(bomb_pos, bomber.id):
            return False, None
        
        bomb_timer = self.client.bomb_delay / 1000.0
        speed = 2 + (self.client.booster_manager.speed_level if self.client.booster_manager else 0)
        
        bomber_in_blast = False
        if bomber.pos.x == bomb_pos.x and abs(bomber.pos.y - bomb_pos.y) <= self.client.bomb_range:
            blocked = False
            step = 1 if bomber.pos.y > bomb_pos.y else -1
            for y in range(bomb_pos.y + step, bomber.pos.y, step):
                if Position(bomb_pos.x, y) in self.client.walls:
                    blocked = True
                    break
            if not blocked:
                bomber_in_blast = True
        elif bomber.pos.y == bomb_pos.y and abs(bomber.pos.x - bomb_pos.x) <= self.client.bomb_range:
            blocked = False
            step = 1 if bomber.pos.x > bomb_pos.x else -1
            for x in range(bomb_pos.x + step, bomber.pos.x, step):
                if Position(x, bomb_pos.y) in self.client.walls:
                    blocked = True
                    break
            if not blocked:
                bomber_in_blast = True
        
        safety_margin = 0.3 if aggressive else 0.5
        min_danger_time = 0.2 if aggressive else 0.5
        if not bomber_in_blast:
            safety_margin = 0.5 if aggressive else 1.0
            min_danger_time = 0.3 if aggressive else 1.0
        
        safe_escape_positions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            escape_pos = Position(bomb_pos.x + dx, bomb_pos.y + dy)
            if not self.client.is_valid_position(escape_pos):
                continue
            
            in_blast_radius = False
            if escape_pos.x == bomb_pos.x and abs(escape_pos.y - bomb_pos.y) <= self.client.bomb_range:
                blocked = False
                step = 1 if escape_pos.y > bomb_pos.y else -1
                for y in range(bomb_pos.y + step, escape_pos.y, step):
                    if Position(bomb_pos.x, y) in self.client.walls:
                        blocked = True
                        break
                if not blocked:
                    in_blast_radius = True
            elif escape_pos.y == bomb_pos.y and abs(escape_pos.x - bomb_pos.x) <= self.client.bomb_range:
                blocked = False
                step = 1 if escape_pos.x > bomb_pos.x else -1
                for x in range(bomb_pos.x + step, escape_pos.x, step):
                    if Position(x, bomb_pos.y) in self.client.walls:
                        blocked = True
                        break
                if not blocked:
                    in_blast_radius = True
            
            if in_blast_radius:
                continue
            
            escape_path = self.client.find_path(bomber.pos, escape_pos, max_steps=15)
            if not escape_path or len(escape_path) < 2:
                if aggressive and bomber.pos.manhattan_distance(escape_pos) == 1:
                    escape_path = [bomber.pos, escape_pos]
                else:
                    continue
            
            time_to_escape = len(escape_path) / speed
            if time_to_escape < bomb_timer - safety_margin:
                danger_time = self.client.get_blast_danger(escape_pos, time_offset=time_to_escape)
                if danger_time > min_danger_time:
                    safe_escape_positions.append((escape_pos, escape_path, time_to_escape))
        
        if safe_escape_positions:
            best_escape = min(safe_escape_positions, key=lambda x: len(x[1]))
            return True, best_escape[1]
        
        return False, None
    
    def find_bomb_position_for_enemy(self, bomber: Bomber, enemy: Enemy) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить врага"""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(enemy.pos.x + dx, enemy.pos.y + dy)
            
            if not self.client.is_valid_position(bomb_pos):
                continue
            
            if bomb_pos.manhattan_distance(enemy.pos) > self.client.bomb_range:
                continue
            
            is_enclosed = self.client.is_bomber_enclosed(bomber)
            can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=is_enclosed)
            if can_place and escape_path:
                return (bomb_pos, escape_path)
        
        return None
    
    def find_bomb_position_for_ghost(self, bomber: Bomber, ghost: Mob) -> Optional[Tuple[Position, List[Position]]]:
        """Найти позицию для бомбы, чтобы убить призрака"""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            bomb_pos = Position(ghost.pos.x + dx, ghost.pos.y + dy)
            
            if not self.client.is_valid_position(bomb_pos):
                continue
            
            if bomb_pos.manhattan_distance(ghost.pos) > self.client.bomb_range:
                continue
            
            is_enclosed = self.client.is_bomber_enclosed(bomber)
            can_place, escape_path = self.can_safely_place_bomb(bomber, bomb_pos, aggressive=is_enclosed)
            if can_place and escape_path:
                return (bomb_pos, escape_path)
        
        return None
