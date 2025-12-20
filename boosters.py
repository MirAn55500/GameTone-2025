"""
Модуль для работы с бустерами (улучшениями)
"""

import time
from typing import Optional, Dict
from models import BoosterType


class BoosterManager:
    """Менеджер для работы с бустерами"""
    
    def __init__(self, get_boosters_func, buy_booster_func):
        self.get_boosters_func = get_boosters_func
        self.buy_booster_func = buy_booster_func
        
        # Статистика улучшений
        self.speed_level = 0
        self.bomb_delay_level = 0
        self.acrobatics_level = 0
        self.bomb_range = 1
        self.bomb_delay = 8000
        
        # Rate limiting
        self.last_booster_429_error = 0
        self.booster_check_skip_until = 0
        self.last_points = 0
    
    def update_stats(self, data: Optional[Dict] = None) -> None:
        """Обновить статистику улучшений из состояния игры"""
        try:
            if data is None:
                data = self.get_boosters_func()
            
            if not data:
                return
            
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
            self.bomb_delay = state.get("bomb_delay", 8000)
            self.bomb_delay_level = (8000 - self.bomb_delay) // 2000
            speed = state.get("speed", 2)
            self.speed_level = max(0, speed - 2)
            
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
    
    def should_buy_booster(self) -> Optional[int]:
        """Логика покупки бустеров"""
        if time.time() < self.booster_check_skip_until:
            return None
        
        try:
            data = self.get_boosters_func()
            if not data:
                self.booster_check_skip_until = time.time() + 10
                return None
            
            state = data.get("state", {})
            if not state:
                return None
            
            points = state.get("points", 0)
            if points < 1:
                return None
            
            # Обновляем статы
            self.update_stats(data)
            
            available = data.get("available", [])
            if not available:
                return None
            
            # НОВЫЙ ПОРЯДОК: 2131424234
            # 1 = дополнительная бомба (POCKETS)
            # 2 = размер бомбы (BOMB_RANGE)
            # 3 = зрение (VISION)
            # 4 = здоровье (ARMOR)
            
            # Статический счетчик для циклического порядка
            if not hasattr(self, '_booster_cycle'):
                self._booster_cycle = 0
            
            # Порядок: 2, 1, 3, 1, 4, 2, 4, 2, 3, 4
            order = [2, 1, 3, 1, 4, 2, 4, 2, 3, 4]
            current_priority = order[self._booster_cycle % len(order)]
            
            # Проверяем текущий приоритет
            if current_priority == 1:  # Дополнительная бомба
                for b in available:
                    if b.get('type') == 'bomb_count' and b.get('cost', 999) <= points:
                        self._booster_cycle += 1
                        return BoosterType.POCKETS
            
            elif current_priority == 2:  # Размер бомбы
                for b in available:
                    if b.get('type') == 'bomb_range' and b.get('cost', 999) <= points:
                        self._booster_cycle += 1
                        return BoosterType.BOMB_RANGE
            
            elif current_priority == 3:  # Зрение
                for b in available:
                    if b.get('type') == 'vision' and b.get('cost', 999) <= points:
                        self._booster_cycle += 1
                        return BoosterType.VISION
            
            elif current_priority == 4:  # Здоровье
                for b in available:
                    if b.get('type') == 'armor' and b.get('cost', 999) <= points:
                        self._booster_cycle += 1
                        return BoosterType.ARMOR
            
            # Если текущий приоритет недоступен, пробуем следующий в порядке
            self._booster_cycle += 1
            return None
            
            return None
            
        except Exception:
            self.booster_check_skip_until = time.time() + 10
            return None
    
    def buy_booster(self, booster_type: int) -> Dict:
        """Купить бустер"""
        return self.buy_booster_func(booster_type)
