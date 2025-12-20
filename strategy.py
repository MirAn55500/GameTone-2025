"""
Модуль стратегии игры согласно новому алгоритму
"""

import time
import random
from typing import List, Dict, Tuple, Optional, Set
from models import Position, Bomber, Bomb, Enemy, Mob


class GameStrategy:
    """Стратегия игры согласно новому алгоритму"""
    
    def __init__(self, game_client):
        self.client = game_client
        self.spawn_iterations = 0
        self.max_spawn_iterations = 10
        self.bomber_groups = []  # Группы по 2 бомбера
        self.last_mob_positions = {}  # Для отслеживания патрулей: mob_id -> (x, y) или None
        self.bomb_avoidance_time = 3.5  # За 3.5 секунды до взрыва (более безопасно)
    
    def is_spawn_phase(self) -> bool:
        """Проверка, находимся ли мы в фазе спавна"""
        return self.spawn_iterations < self.max_spawn_iterations
    
    def increment_spawn_iterations(self):
        """Увеличить счетчик итераций спавна"""
        self.spawn_iterations += 1
    
    def check_bomb_avoidance(self, bomber: Bomber) -> Optional[List[Position]]:
        """Активное избегание бомб - убегаем заранее (за 3-4 секунды)"""
        danger_timer = self.client.get_blast_danger(bomber.pos)
        
        # Увеличиваем время избегания до 3-4 секунд для большей безопасности
        avoidance_threshold = 3.5
        
        # Если опасность через 3.5 секунды или меньше - активно убегаем
        if danger_timer <= avoidance_threshold:
            # Ищем безопасную позицию (только горизонтальные/вертикальные шаги!)
            best_escape = None
            max_safety = -1
            escape_path = []
            
            # Проверяем все соседние позиции
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                if not self.client.is_valid_position(escape_pos):
                    continue
                
                # Проверяем безопасность текущей позиции
                safety_time = self.client.get_blast_danger(escape_pos)
                
                # Проверяем безопасность через время прибытия (скорость ~2 клетки/сек)
                arrival_time = 0.5  # Время до прибытия на соседнюю клетку
                future_safety = self.client.get_blast_danger(escape_pos, time_offset=arrival_time)
                
                # Берем минимум из текущей и будущей безопасности
                min_safety = min(safety_time, future_safety)
                
                if min_safety > max_safety:
                    max_safety = min_safety
                    best_escape = escape_pos
            
            if best_escape and max_safety > avoidance_threshold:
                # Возвращаем путь убегания (можем сделать несколько шагов)
                escape_path = [bomber.pos, best_escape]
                
                # Пробуем продолжить путь убегания для большей безопасности
                current = best_escape
                for _ in range(2):  # Делаем еще 2 шага от опасности
                    best_next = None
                    best_next_safety = -1
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        next_pos = Position(current.x + dx, current.y + dy)
                        if not self.client.is_valid_position(next_pos):
                            continue
                        if next_pos in escape_path:  # Не возвращаемся
                            continue
                        
                        # Проверяем безопасность с учетом времени прибытия
                        arrival_time = (len(escape_path)) * 0.5
                        safety = self.client.get_blast_danger(next_pos, time_offset=arrival_time)
                        if safety > best_next_safety:
                            best_next_safety = safety
                            best_next = next_pos
                    
                    if best_next and best_next_safety > avoidance_threshold:
                        escape_path.append(best_next)
                        current = best_next
                    else:
                        break
                
                return escape_path
        
        return None
    
    def handle_mob(self, bomber: Bomber, mob: Mob) -> Optional[Dict]:
        """Обработка моба согласно алгоритму"""
        if mob.type == "ghost":
            # Призрак: останавливаемся на безопасном расстоянии (3-4 клетки) и ставим бомбу
            distance = bomber.pos.manhattan_distance(mob.pos)
            safe_distance_min = 3  # Минимальное безопасное расстояние
            safe_distance_max = 4  # Максимальное безопасное расстояние для атаки
            
            if distance < safe_distance_min:
                # Слишком близко - отходим на безопасное расстояние
                # Вычисляем направление от призрака
                dx = bomber.pos.x - mob.pos.x
                dy = bomber.pos.y - mob.pos.y
                
                # Нормализуем направление
                if dx != 0:
                    move_dx = 1 if dx > 0 else -1
                else:
                    move_dx = 0
                    
                if dy != 0:
                    move_dy = 1 if dy > 0 else -1
                else:
                    move_dy = 0
                
                # Делаем шаг от призрака
                escape_pos = Position(bomber.pos.x + move_dx, bomber.pos.y + move_dy)
                if self.client.is_valid_position(escape_pos):
                    return {
                        "id": bomber.id,
                        "path": [[escape_pos.x, escape_pos.y]],
                        "bombs": []
                    }
                else:
                    # Если не можем отойти в этом направлении, пробуем другие
                    for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                        if self.client.is_valid_position(escape_pos):
                            return {
                                "id": bomber.id,
                                "path": [[escape_pos.x, escape_pos.y]],
                                "bombs": []
                            }
            
            elif safe_distance_min <= distance <= safe_distance_max:
                # На безопасном расстоянии - ставим бомбу и отходим
                if bomber.bombs_available > 0:
                    bomb_pos = bomber.pos
                    
                    # Проверяем, может ли бомба достать до призрака
                    can_reach = False
                    for r_dx, r_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        for r in range(1, self.client.bomb_range + 1):
                            check_pos = Position(bomb_pos.x + r_dx * r, bomb_pos.y + r_dy * r)
                            if check_pos == mob.pos:
                                can_reach = True
                                break
                            if check_pos in self.client.walls:
                                break
                        if can_reach:
                            break
                    
                    if can_reach:
                        # Отходим от призрака после установки бомбы
                        dx = bomber.pos.x - mob.pos.x
                        dy = bomber.pos.y - mob.pos.y
                        
                        # Выбираем направление от призрака
                        move_dx, move_dy = 0, 0
                        if abs(dx) > abs(dy):
                            move_dx = 1 if dx > 0 else -1
                        else:
                            move_dy = 1 if dy > 0 else -1
                        
                        escape_pos = Position(bomber.pos.x + move_dx, bomber.pos.y + move_dy)
                        if self.client.is_valid_position(escape_pos):
                            return {
                                "id": bomber.id,
                                "bombs": [[bomb_pos.x, bomb_pos.y]],
                                "path": [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                            }
                        else:
                            # Пробуем другие направления
                            for edx, edy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                escape_pos = Position(bomber.pos.x + edx, bomber.pos.y + edy)
                                if self.client.is_valid_position(escape_pos):
                                    return {
                                        "id": bomber.id,
                                        "bombs": [[bomb_pos.x, bomb_pos.y]],
                                        "path": [[bomber.pos.x, bomber.pos.y], [escape_pos.x, escape_pos.y]]
                                    }
            
            elif distance > safe_distance_max:
                # Слишком далеко - приближаемся до безопасного расстояния
                dx = mob.pos.x - bomber.pos.x
                dy = mob.pos.y - bomber.pos.y
                
                # Выбираем направление к призраку (но не подходим слишком близко)
                move_dx, move_dy = 0, 0
                if abs(dx) > abs(dy):
                    move_dx = 1 if dx > 0 else -1
                else:
                    move_dy = 1 if dy > 0 else -1
                
                move_pos = Position(bomber.pos.x + move_dx, bomber.pos.y + move_dy)
                if self.client.is_valid_position(move_pos):
                    new_dist = move_pos.manhattan_distance(mob.pos)
                    # Приближаемся только если не станем слишком близко
                    if new_dist >= safe_distance_min:
                        return {
                            "id": bomber.id,
                            "path": [[move_pos.x, move_pos.y]],
                            "bombs": []
                        }
            
            # Если ничего не подошло, просто стоим
            return {"id": bomber.id, "path": [[bomber.pos.x, bomber.pos.y]], "bombs": []}
        elif mob.type == "patrol":
            # Патруль: найти неподвижную координату и поставить бомбу на текущей позиции
            mob_id = mob.id
            if mob_id in self.last_mob_positions:
                last_pos = self.last_mob_positions[mob_id]
                if last_pos == (mob.pos.x, mob.pos.y):
                    # Координата не изменилась - это неподвижная координата
                    if bomber.bombs_available > 0:
                        # Ставим бомбу на текущей позиции бомбера
                        bomb_pos = bomber.pos
                        return {
                            "id": bomber.id,
                            "bombs": [[bomb_pos.x, bomb_pos.y]],
                            "path": [[bomber.pos.x, bomber.pos.y]]
                        }
            
            # Обновляем позицию патруля
            self.last_mob_positions[mob_id] = (mob.pos.x, mob.pos.y)
        
        return None
    
    def _create_square_path_around_bomb(self, bomb_pos: Position) -> Optional[List[Position]]:
        """Создать квадратный путь вокруг бомбы (только горизонтальные/вертикальные шаги!)"""
        square_path = []
        # Создаем квадрат: вправо, вниз, влево, вверх
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        current_pos = bomb_pos
        
        # Делаем один полный оборот (4 шага)
        for dx, dy in directions:
            next_pos = Position(current_pos.x + dx, current_pos.y + dy)
            if self.client.is_valid_position(next_pos):
                square_path.append(next_pos)
                current_pos = next_pos
            else:
                # Если не можем идти в этом направлении, пробуем следующее
                continue
        
        # Если получили хотя бы один шаг - возвращаем
        return square_path if len(square_path) > 0 else None
    
    def handle_enemy_bomber(self, bomber: Bomber, enemy: Enemy) -> Optional[Dict]:
        """Обработка вражеского бомбера: ставим бомбу на текущей позиции если враг близко"""
        if bomber.bombs_available == 0:
            # Нет бомбы - ждем
            return {"id": bomber.id, "path": [[bomber.pos.x, bomber.pos.y]], "bombs": []}
        
        distance = bomber.pos.manhattan_distance(enemy.pos)
        bomb_pos = bomber.pos  # Бомба ставится только на текущей позиции!
        
        # Если враг очень близко (расстояние 0-2) - ВСЕГДА ставим бомбу
        if distance <= 2:
            # Ставим бомбу на текущей позиции независимо от других бомберов
            # (приоритет - убить врага)
            return {
                "id": bomber.id,
                "bombs": [[bomb_pos.x, bomb_pos.y]],
                "path": [[bomber.pos.x, bomber.pos.y]]
            }
        
        # Проверяем, может ли бомба достать до врага
        can_reach_enemy = False
        for r_dx, r_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for r in range(1, self.client.bomb_range + 1):
                check_pos = Position(bomb_pos.x + r_dx * r, bomb_pos.y + r_dy * r)
                if check_pos == enemy.pos:
                    can_reach_enemy = True
                    break
                if check_pos in self.client.walls:
                    break
            if can_reach_enemy:
                break
        
        if can_reach_enemy:
            # Бомба достанет до врага - ставим на текущей позиции
            # Проверяем, не попадет ли в других бомберов (но только если враг не очень близко)
            will_hit_others = self.client.will_bomb_hit_other_bombers(bomb_pos, bomber.id)
            if not will_hit_others:
                return {
                    "id": bomber.id,
                    "bombs": [[bomb_pos.x, bomb_pos.y]],
                    "path": [[bomber.pos.x, bomber.pos.y]]
                }
            # Если попадет в других, но враг близко - все равно ставим
            elif distance <= 3:
                return {
                    "id": bomber.id,
                    "bombs": [[bomb_pos.x, bomb_pos.y]],
                    "path": [[bomber.pos.x, bomber.pos.y]]
                }
        
        # Если враг слишком далеко - делаем 1-2 шага навстречу (только горизонтальные/вертикальные!)
        dx = enemy.pos.x - bomber.pos.x
        dy = enemy.pos.y - bomber.pos.y
        
        # Выбираем направление (приоритет X, потом Y) - только одно направление за раз!
        move_dx, move_dy = 0, 0
        if dx != 0:
            move_dx = 1 if dx > 0 else -1
        elif dy != 0:
            move_dy = 1 if dy > 0 else -1
        
        # Делаем первый шаг
        step1 = Position(bomber.pos.x + move_dx, bomber.pos.y + move_dy)
        if self.client.is_valid_position(step1):
            # Делаем второй шаг в том же направлении
            step2 = Position(step1.x + move_dx, step1.y + move_dy)
            if self.client.is_valid_position(step2):
                # Два шага навстречу врагу
                return {
                    "id": bomber.id,
                    "path": [[step1.x, step1.y], [step2.x, step2.y]],
                    "bombs": []
                }
            else:
                # Только один шаг
                return {
                    "id": bomber.id,
                    "path": [[step1.x, step1.y]],
                    "bombs": []
                }
        
        return None
    
    def _create_path_avoiding_bomb_coords(self, bomber: Bomber, bomb_pos: Position, 
                                         initial_escape: List[Position]) -> List[Position]:
        """Создать путь так, чтобы через время взрыва координаты не совпадали с бомбой"""
        bomb_timer = self.client.bomb_delay / 1000.0
        speed = 2 + (self.client.booster_manager.speed_level if self.client.booster_manager else 0)
        
        # Вычисляем позицию через время взрыва
        steps_until_explosion = max(1, int(bomb_timer * speed))
        
        # Берем начальный путь escape
        path = initial_escape[:] if initial_escape else [bomber.pos]
        
        # Если путь слишком короткий, дополняем его так, чтобы позиция через время взрыва
        # имела координаты, отличные от бомбы
        if len(path) < steps_until_explosion:
            current_pos = path[-1] if path else bomber.pos
            for _ in range(steps_until_explosion - len(path)):
                # Ищем позицию, где x != bomb_pos.x И y != bomb_pos.y
                best_next = None
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_pos = Position(current_pos.x + dx, current_pos.y + dy)
                    if not self.client.is_valid_position(next_pos):
                        continue
                    if next_pos.x != bomb_pos.x and next_pos.y != bomb_pos.y:
                        best_next = next_pos
                        break
                
                if best_next:
                    path.append(best_next)
                    current_pos = best_next
                else:
                    # Если не можем найти подходящую позицию, просто идем дальше от бомбы
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        next_pos = Position(current_pos.x + dx, current_pos.y + dy)
                        if self.client.is_valid_position(next_pos):
                            path.append(next_pos)
                            current_pos = next_pos
                            break
                    else:
                        break
        
        return path
    
    def handle_spawn_phase(self, alive_bombers: List[Bomber]) -> Optional[Dict]:
        """Обработка фазы спавна"""
        if not alive_bombers:
            return None
        
        first_bomber = alive_bombers[0]
        has_exit = self.client.check_exit_within_radius(first_bomber.pos, radius=5)
        all_enclosed = self.client.are_all_bombers_enclosed()
        
        commands = []
        
        if not has_exit or all_enclosed:
            # Замкнутое пространство - идем к ближайшей стене всеми
            nearest_wall = self._find_nearest_wall(first_bomber.pos)
            if nearest_wall:
                for bomber in alive_bombers:
                    if not bomber.can_move:
                        continue
                    path = self.client.find_path(bomber.pos, nearest_wall, max_steps=10)
                    if path:
                        commands.append({
                            "id": bomber.id,
                            "path": [[p.x, p.y] for p in path[:10]],
                            "bombs": []
                        })
        else:
            # Открытое пространство - отправляем 4/6 бомберов
            if len(alive_bombers) >= 6:
                exit_bombers = alive_bombers[:4]
                stay_bombers = alive_bombers[4:6]
            else:
                exit_bombers = alive_bombers[:len(alive_bombers) * 2 // 3]
                stay_bombers = alive_bombers[len(exit_bombers):]
            
            # Отправляем группу к открытому пространству
            for bomber in exit_bombers:
                if not bomber.can_move:
                    continue
                open_space = self._find_open_space(bomber.pos)
                if open_space:
                    path = self.client.find_path(bomber.pos, open_space, max_steps=15)
                    if path:
                        commands.append({
                            "id": bomber.id,
                            "path": [[p.x, p.y] for p in path[:15]],
                            "bombs": []
                        })
            
            # Остальные остаются ломать стены
            for bomber in stay_bombers:
                if not bomber.can_move:
                    continue
                nearest_wall = self._find_nearest_wall(bomber.pos)
                if nearest_wall:
                    path = self.client.find_path(bomber.pos, nearest_wall, max_steps=10)
                    if path:
                        commands.append({
                            "id": bomber.id,
                            "path": [[p.x, p.y] for p in path[:10]],
                            "bombs": []
                        })
        
        if commands:
            return self.client.move_bombers(commands)
        
        return None
    
    def _find_nearest_wall(self, pos: Position) -> Optional[Position]:
        """Найти ближайшую разрушаемую стену или препятствие"""
        nearest = None
        min_dist = float('inf')
        
        # Сначала ищем препятствия
        for obs in self.client.obstacles:
            dist = pos.manhattan_distance(obs)
            if dist < min_dist:
                min_dist = dist
                nearest = obs
        
        # Если нет препятствий, ищем ближайшую стену (неразрушаемую)
        if nearest is None:
            for wall in self.client.walls:
                dist = pos.manhattan_distance(wall)
                if dist < min_dist:
                    min_dist = dist
                    nearest = wall
        
        return nearest
    
    def _find_open_space(self, pos: Position) -> Optional[Position]:
        """Найти открытое пространство"""
        # Ищем свободную клетку без стен в радиусе
        for radius in range(5, 15):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    check_pos = Position(pos.x + dx, pos.y + dy)
                    if not self.client.is_valid_position(check_pos):
                        continue
                    if check_pos not in self.client.walls and check_pos not in self.client.obstacles:
                        return check_pos
        return None
    
    def organize_groups(self, alive_bombers: List[Bomber]):
        """Организовать группы - каждый бомбер действует независимо"""
        # Каждый бомбер в своей группе (не по парам)
        self.bomber_groups = [[bomber] for bomber in alive_bombers]
    
    def handle_single_bomber(self, bomber: Bomber) -> Optional[Dict]:
        """Обработка одного оставшегося бомбера - АКТИВНОЕ движение с бомбовым следом"""
        if not bomber.alive or not bomber.can_move:
            return None
        
        # Ищем лучшую позицию для движения (безопасную и далекую)
        best_move = None
        best_score = -1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            next_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
            if not self.client.is_valid_position(next_pos):
                continue
            if next_pos in self.client.walls:
                continue
            
            # Проверяем безопасность
            danger = self.client.get_blast_danger(next_pos, time_offset=0.5)
            
            # Предпочитаем позиции с препятствиями рядом (для установки бомб)
            nearby_obstacles = [obs for obs in self.client.obstacles 
                              if next_pos.manhattan_distance(obs) <= self.client.bomb_range + 2]
            
            # Счет: безопасность + наличие препятствий
            score = danger + len(nearby_obstacles) * 10
            
            if score > best_score:
                best_score = score
                best_move = next_pos
        
        if best_move:
            path = [bomber.pos, best_move]
            bombs = []
            
            # АКТИВНО: Ставим бомбы при движении (бомбовый след)
            if bomber.bombs_available > 0:
                current_danger = self.client.get_blast_danger(bomber.pos)
                if current_danger > 5.0:  # Текущая позиция безопасна
                    will_hit_others = self.client.will_bomb_hit_other_bombers(bomber.pos, bomber.id)
                    if not will_hit_others:
                        # Проверяем, можем ли убежать
                        move_danger = self.client.get_blast_danger(best_move, time_offset=0.5)
                        if move_danger > 4.0:
                            bombs = [[bomber.pos.x, bomber.pos.y]]
                            # Продолжаем путь убегания
                            can_place, escape_path = self.client.can_safely_place_bomb(bomber, bomber.pos, aggressive=True)
                            if escape_path and len(escape_path) > 1:
                                path = [bomber.pos] + escape_path[:4]
            
            return {
                "id": bomber.id,
                "path": [[p.x, p.y] for p in path],
                "bombs": bombs
            }
        
        # Если не можем двигаться, хотя бы ставим бомбу если есть
        if bomber.bombs_available > 0:
            return {
                "id": bomber.id,
                "path": [[bomber.pos.x, bomber.pos.y]],
                "bombs": [[bomber.pos.x, bomber.pos.y]]
            }
        
        return {"id": bomber.id, "path": [[bomber.pos.x, bomber.pos.y]], "bombs": []}
    
    def find_best_wall_target(self, bomber: Bomber, assigned_targets: Set) -> Optional[Position]:
        """Найти лучшую стену для разрушения - учитываем расхождение бомберов"""
        if not self.client.obstacles:
            # Если нет препятствий, ищем ближайшую стену или открытое пространство
            return self._find_nearest_wall(bomber.pos) or self._find_open_space(bomber.pos)
        
        # Находим центр всех живых бомберов для определения направления расхождения
        alive_bombers = [b for b in self.client.bombers if b.alive]
        if len(alive_bombers) > 1:
            center_x = sum(b.pos.x for b in alive_bombers) // len(alive_bombers)
            center_y = sum(b.pos.y for b in alive_bombers) // len(alive_bombers)
            center_pos = Position(center_x, center_y)
        else:
            center_pos = bomber.pos
        
        # Определяем направление от центра к текущему бомберу
        dx_from_center = bomber.pos.x - center_pos.x
        dy_from_center = bomber.pos.y - center_pos.y
        
        # Определяем предпочтительное направление (квадрант)
        preferred_direction = None
        if abs(dx_from_center) > abs(dy_from_center):
            preferred_direction = "east" if dx_from_center > 0 else "west"
        else:
            preferred_direction = "south" if dy_from_center > 0 else "north"
        
        best_target = None
        best_score = -1
        
        for obs in self.client.obstacles:
            # Проверяем, назначена ли цель
            if (obs.x, obs.y) in assigned_targets:
                continue
            
            # Подсчитываем, сколько стен можно разрушить одной бомбой
            wall_score = self._count_destroyable_walls(obs)
            
            # Расстояние от текущего бомбера
            distance_from_bomber = bomber.pos.manhattan_distance(obs)
            
            # Расстояние от других бомберов (предпочитаем цели дальше от других)
            min_dist_to_others = float('inf')
            for other in alive_bombers:
                if other.id == bomber.id:
                    continue
                dist = other.pos.manhattan_distance(obs)
                min_dist_to_others = min(min_dist_to_others, dist)
            
            # Проверяем, находится ли цель в предпочтительном направлении
            obs_dx = obs.x - center_pos.x
            obs_dy = obs.y - center_pos.y
            direction_match = False
            if preferred_direction == "east" and obs_dx > 0:
                direction_match = True
            elif preferred_direction == "west" and obs_dx < 0:
                direction_match = True
            elif preferred_direction == "south" and obs_dy > 0:
                direction_match = True
            elif preferred_direction == "north" and obs_dy < 0:
                direction_match = True
            
            # Комплексный счет: количество стен * 100 + расстояние от других * 10 + бонус за направление
            score = wall_score * 100 + min_dist_to_others * 10
            if direction_match:
                score += 50  # Бонус за направление расхождения
            
            # Небольшой штраф за слишком большое расстояние от бомбера
            if distance_from_bomber > 20:
                score -= 20
            
            if score > best_score:
                best_score = score
                best_target = obs
        
        # Если все препятствия назначены, ищем ближайшее не назначенное
        if best_target is None:
            for obs in self.client.obstacles:
                if (obs.x, obs.y) not in assigned_targets:
                    return obs
        
        return best_target
    
    def _count_destroyable_walls(self, center: Position) -> int:
        """Подсчитать количество стен, которые можно разрушить одной бомбой в этой позиции"""
        count = 0
        bomb_range = self.client.bomb_range
        
        # Проверяем все направления
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for r in range(1, bomb_range + 1):
                check_pos = Position(center.x + dx * r, center.y + dy * r)
                if check_pos in self.client.obstacles:
                    count += 1
                elif check_pos in self.client.walls:
                    break  # Стена останавливает взрыв
        
        return count
    
