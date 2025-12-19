"""
SquadManager - Координация отряда юнитов для предотвращения deadlock
Решает проблемы:
1. "Каша" - несколько юнитов на одной клетке
2. Deadlock - юниты не могут поставить бомбу из-за проверки безопасности
3. Конфликты целей - юниты идут в одну и ту же клетку
"""

import heapq
from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from game_client import Position, Bomber


@dataclass
class UnitMove:
    """Команда движения для юнита"""
    action: str  # 'MOVE', 'BOMB', 'WAIT', 'SCATTER'
    target: Optional[Tuple[int, int]] = None
    bomb_pos: Optional[Tuple[int, int]] = None
    path: Optional[List[Tuple[int, int]]] = None


class SquadManager:
    """Менеджер отряда - координирует действия всех юнитов"""
    
    def __init__(self, game_client):
        self.client = game_client
    
    def count_mates_nearby(self, bomber: 'Bomber', radius: int = 1) -> int:
        """Подсчитывает количество союзников рядом с юнитом"""
        count = 0
        for other in self.client.bombers:
            if other.id == bomber.id or not other.alive:
                continue
            dist = bomber.pos.manhattan_distance(other.pos)
            if dist <= radius:
                count += 1
        return count
    
    def get_units_on_tile(self, pos: 'Position') -> List['Bomber']:
        """Возвращает список юнитов на указанной клетке"""
        units = []
        for bomber in self.client.bombers:
            if bomber.alive and bomber.pos == pos:
                units.append(bomber)
        return units
    
    def scatter_move(self, bomber: 'Bomber', occupied_targets: Set['Position'], max_distance: int = 5) -> Optional[UnitMove]:
        """
        Находит ближайшую свободную клетку для рассеивания.
        Используется когда юниты слиплись в "кашу".
        """
        start = bomber.pos
        queue = [(0, start)]  # (distance, position)
        visited = {start}
        
        # Получаем позиции всех других юнитов
        other_positions = {b.pos for b in self.client.bombers if b.id != bomber.id and b.alive}
        
        # Импортируем Position локально, чтобы избежать циклических зависимостей
        from game_client import Position
        
        while queue:
            dist, current = heapq.heappop(queue)
            
            if dist > max_distance:
                continue
            
            # Проверяем, что клетка свободна и проходима
            if (current not in occupied_targets and 
                current not in other_positions and
                self.client.is_valid_position(current)):
                
                # Проверяем безопасность
                danger = self.client.get_blast_danger(current)
                if danger > 0.5:  # Минимальная безопасность
                    # Нашли свободную клетку
                    path = self.client.find_path(bomber.pos, current, max_steps=dist + 2)
                    if path and len(path) > 1:
                        return UnitMove(
                            action='MOVE',
                            target=current,
                            path=[(p.x, p.y) for p in path]
                        )
            
            # Добавляем соседей в очередь
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                if neighbor not in visited:
                    visited.add(neighbor)
                    if self.client.is_valid_position(neighbor):
                        heapq.heappush(queue, (dist + 1, neighbor))
        
        return None
    
    def is_safe_for_team(self, bomb_pos: 'Position', bomber_id: str) -> bool:
        """
        Проверяет, можно ли безопасно поставить бомбу для команды.
        Учитывает, что союзники могут убежать.
        """
        # Получаем зону взрыва
        danger_cells = self._get_explosion_cells(bomb_pos)
        
        # Проверяем каждого союзника
        for bomber in self.client.bombers:
            if bomber.id == bomber_id or not bomber.alive:
                continue
            
            # Если союзник в зоне взрыва
            if bomber.pos in danger_cells:
                # Проверяем, может ли он убежать
                if not self._can_escape_from_bomb(bomber, bomb_pos, danger_cells):
                    return False
        
        return True
    
    def _get_explosion_cells(self, bomb_pos: 'Position') -> Set['Position']:
        """Возвращает множество клеток, которые заденет взрыв"""
        from game_client import Position
        
        cells = {bomb_pos}
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in dirs:
            for r in range(1, self.client.bomb_range + 1):
                nx, ny = bomb_pos.x + dx * r, bomb_pos.y + dy * r
                cell = Position(nx, ny)
                
                # Стены останавливают взрыв
                if cell in self.client.walls:
                    break
                
                cells.add(cell)
                
                # Ящики останавливают взрыв
                if cell in self.client.obstacles:
                    break
        
        return cells
    
    def _can_escape_from_bomb(self, bomber: 'Bomber', bomb_pos: 'Position', danger_cells: Set['Position']) -> bool:
        """Проверяет, может ли юнит убежать от бомбы"""
        from game_client import Position
        
        bomb_timer = self.client.bomb_delay / 1000.0  # секунды
        speed = 2 + self.client.speed_level
        
        # Проверяем соседние клетки
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            escape_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
            
            if not self.client.is_valid_position(escape_pos):
                continue
            
            # Проверяем, что escape позиция не в зоне взрыва
            if escape_pos in danger_cells:
                continue
            
            # Проверяем путь к escape позиции
            escape_path = self.client.find_path(bomber.pos, escape_pos, max_steps=10)
            if escape_path and len(escape_path) > 1:
                time_to_escape = len(escape_path) / speed
                if time_to_escape < bomb_timer - 0.5:  # Запас безопасности
                    # Проверяем, что escape позиция безопасна от других бомб
                    danger = self.client.get_blast_danger(escape_pos, time_offset=time_to_escape)
                    if danger > 1.0:
                        return True
        
        return False
    
    def get_desperate_breakthrough_leader(self, units_on_tile: List['Bomber']) -> Optional['Bomber']:
        """
        Выбирает лидера для отчаянного прорыва.
        Лидер - юнит с минимальным ID (или другим критерием).
        """
        if not units_on_tile:
            return None
        
        # Сортируем по ID и выбираем первого
        sorted_units = sorted(units_on_tile, key=lambda u: u.id)
        return sorted_units[0]
    
    def coordinate_moves(self, bombers: List['Bomber'], assigned_targets: Set['Position']) -> Dict[str, UnitMove]:
        """
        Координирует движения всех юнитов.
        Возвращает словарь {bomber_id: UnitMove}
        """
        moves = {}
        occupied_targets = set(assigned_targets)  # Клетки, куда уже идут другие юниты
        planned_bomb_positions = {}  # bomber_id -> bomb_pos
        
        # Сортируем юнитов: сначала те, кто в опасности
        def danger_priority(bomber):
            danger = self.client.get_blast_danger(bomber.pos)
            return 0.0 if danger < float('inf') else 1000.0
        
        sorted_bombers = sorted(bombers, key=danger_priority)
        
        for bomber in sorted_bombers:
            if not bomber.alive or not bomber.can_move:
                continue
            
            # ШАГ 1: Проверка на "кашу" (Anti-Clump)
            mates_nearby = self.count_mates_nearby(bomber, radius=1)
            units_on_tile = self.get_units_on_tile(bomber.pos)
            
            if mates_nearby > 1:
                # Включаем режим рассеивания
                scatter_move = self.scatter_move(bomber, occupied_targets)
                if scatter_move:
                    moves[bomber.id] = scatter_move
                    if scatter_move.target:
                        occupied_targets.add(scatter_move.target)
                    print(f"[SQUAD] {bomber.id} scattering from clump at {bomber.pos} (mates={mates_nearby})")
                    continue
                else:
                    # Не можем рассеяться - проверяем отчаянный прорыв
                    if bomber.bombs_available > 0:
                        leader = self.get_desperate_breakthrough_leader(units_on_tile)
                        if leader and leader.id == bomber.id:
                            # Лидер ставит бомбу для прорыва
                            is_enclosed = self.client.is_bomber_enclosed(bomber)
                            if is_enclosed:
                                # Отчаянный прорыв - ставим бомбу даже если опасно для команды
                                moves[bomber.id] = UnitMove(
                                    action='BOMB',
                                    bomb_pos=bomber.pos,
                                    target=bomber.pos
                                )
                                print(f"[SQUAD] {bomber.id} DESPERATE BREAKTHROUGH - planting bomb at {bomber.pos}")
                                continue
                        else:
                            # Не лидер - пытаемся отойти
                            from game_client import Position
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                move_pos = Position(bomber.pos.x + dx, bomber.pos.y + dy)
                                if (self.client.is_valid_position(move_pos) and 
                                    move_pos not in occupied_targets):
                                    moves[bomber.id] = UnitMove(
                                        action='MOVE',
                                        target=move_pos,
                                        path=[(bomber.pos.x, bomber.pos.y), (move_pos.x, move_pos.y)]
                                    )
                                    occupied_targets.add(move_pos)
                                    print(f"[SQUAD] {bomber.id} moving away from clump (not leader)")
                                    break
                            continue
            
            # ШАГ 2: Обычная логика (будет обработана в make_move)
            # Здесь мы только координируем, основная логика остается в make_move
            # Но мы можем пометить занятые цели
            moves[bomber.id] = UnitMove(action='CONTINUE')  # Продолжить обычную логику
        
        return moves
