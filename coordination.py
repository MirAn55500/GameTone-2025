"""
Модуль для координации бомберов
"""

from typing import List, Set, Optional
from models import Position, Bomber


class BomberCoordinator:
    """Класс для координации действий бомберов"""
    
    def __init__(self, check_exit_func, get_collective_targets_func):
        self.check_exit_func = check_exit_func
        self.get_collective_targets_func = get_collective_targets_func
    
    def are_bombers_at_same_point(self, bombers: List[Bomber], tolerance: int = 1) -> bool:
        """Проверить, находятся ли все бомберы в одной точке (с допуском)"""
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
    
    def get_bomber_groups(self, bombers: List[Bomber], max_distance: int = 3) -> List[List[Bomber]]:
        """Разделить бомберов на группы по близости"""
        groups = []
        used = set()
        
        for bomber in bombers:
            if bomber.id in used or not bomber.alive:
                continue
            
            group = [bomber]
            used.add(bomber.id)
            
            for other in bombers:
                if other.id in used or not other.alive:
                    continue
                if bomber.pos.manhattan_distance(other.pos) <= max_distance:
                    group.append(other)
                    used.add(other.id)
            
            if len(group) > 0:
                groups.append(group)
        
        return groups
    
    def are_all_bombers_enclosed(self, bombers: List[Bomber], obstacles: Set[Position],
                                 bomb_range: int, check_exit_func) -> bool:
        """Проверить, заперты ли все бомберы"""
        alive_bombers = [b for b in bombers if b.alive]
        if not alive_bombers:
            return False
        
        first_bomber = alive_bombers[0]
        has_exit = check_exit_func(first_bomber.pos, radius=5)
        
        if not has_exit and len(obstacles) > 0:
            for obs in obstacles:
                dist = first_bomber.pos.manhattan_distance(obs)
                if dist <= bomb_range + 3:
                    return True
        
        return not has_exit
    
    def get_collective_targets(self, bombers: List[Bomber]) -> List[Position]:
        """Получить коллективные цели для группы бомберов"""
        return self.get_collective_targets_func(bombers)
