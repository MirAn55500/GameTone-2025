"""
Модели данных для игры Bomberman
"""

import math
from typing import Optional
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
