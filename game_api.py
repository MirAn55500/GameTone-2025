"""
Общий модуль для работы с API игры
Используется и клиентом, и визуализатором
"""

import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MapAnalysis:
    """Анализ карты для улучшения стратегии"""
    obstacle_density: float  # Плотность препятствий
    obstacle_clusters: List[List[Tuple[int, int]]]  # Кластеры препятствий
    safe_zones: List[Tuple[int, int]]  # Безопасные зоны
    high_value_targets: List[Tuple[int, int]]  # Высокоценные цели (группы препятствий)
    map_size: Tuple[int, int]
    walls: List[Tuple[int, int]]
    obstacles: List[Tuple[int, int]]


class GameAPI:
    """Общий класс для работы с API игры с кэшированием"""
    
    def __init__(self, api_key: str, base_url: str = "https://games.datsteam.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})
        
        # Кэш данных
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 0.35  # TTL кэша в секундах (чуть меньше чем частота обновления)
        
        # Rate limiting
        self.request_times = []
        self.last_429_error = 0
        self.backoff_until = 0
        
        # Анализ карты
        self.map_analysis: Optional[MapAnalysis] = None
        self.last_analysis_time = 0
    
    def _rate_limit(self):
        """Ограничение скорости запросов (3 в секунду)"""
        now = time.time()
        request_times = [t for t in self.request_times if now - t < 1.0]

        if len(request_times) >= 3:
            sleep_time = 1.0 - (now - request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.request_times.append(time.time())
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Получить данные из кэша"""
        if key in self.cache and key in self.cache_time:
            if time.time() - self.cache_time[key] < self.cache_ttl:
                return self.cache[key]
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Сохранить данные в кэш"""
        self.cache[key] = data
        self.cache_time[key] = time.time()
    
    def get_arena(self, use_cache: bool = True) -> Dict:
        """Получить состояние арены с кэшированием"""
        cache_key = "arena"
        
        # Проверяем кэш
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        # Проверяем backoff после ошибки 429
        if time.time() < self.backoff_until:
            # Возвращаем кэшированные данные, если есть
            if cache_key in self.cache:
                return self.cache[cache_key]
            raise Exception("Rate limited, waiting...")
        
        try:
            self._rate_limit()
            response = self.session.get(f"{self.base_url}/api/arena", timeout=5)
            
            if response.status_code == 429:
                self.last_429_error = time.time()
                self.backoff_until = time.time() + 2.0  # Ждем 2 секунды
                if cache_key in self.cache:
                    return self.cache[cache_key]
                raise Exception("Too many requests")
            
            response.raise_for_status()
            data = response.json()

            # Логируем очки, если они есть в ответе arena
            if "points" in data:
                print(f"[SCORE] Arena points: {data['points']}")

            # Сохраняем в кэш
            self._set_cache(cache_key, data)

            # Сбрасываем backoff при успехе
            self.backoff_until = 0

            return data
        except requests.exceptions.RequestException as e:
            # При ошибке возвращаем кэш, если есть
            if cache_key in self.cache:
                return self.cache[cache_key]
            raise
    
    def analyze_map(self, data: Dict) -> MapAnalysis:
        """Анализировать карту для улучшения стратегии"""
        arena = data.get("arena", {})
        walls = [tuple(w) for w in arena.get("walls", [])]
        obstacles = [tuple(o) for o in arena.get("obstacles", [])]
        map_size = tuple(data.get("map_size", [30, 30]))
        
        # Плотность препятствий
        total_cells = map_size[0] * map_size[1]
        obstacle_density = len(obstacles) / total_cells if total_cells > 0 else 0
        
        # Находим кластеры препятствий (группы рядом стоящих)
        obstacle_set = set(obstacles)
        clusters = []
        visited = set()
        
        for obstacle in obstacles:
            if obstacle in visited:
                continue
            
            # BFS для поиска кластера
            cluster = []
            queue = [obstacle]
            visited.add(obstacle)
            
            while queue:
                x, y = queue.pop(0)
                cluster.append((x, y))
                
                # Проверяем соседей
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in obstacle_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(cluster) > 1:  # Только кластеры из 2+ препятствий
                clusters.append(cluster)
        
        # Находим высокоценные цели (кластеры, которые можно уничтожить одной бомбой)
        high_value_targets = []
        for cluster in clusters:
            if len(cluster) >= 2:
                # Находим центр кластера или позицию для бомбы
                center_x = sum(x for x, y in cluster) // len(cluster)
                center_y = sum(y for x, y in cluster) // len(cluster)
                
                # Ищем позицию рядом с центром для размещения бомбы
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                    bomb_pos = (center_x + dx, center_y + dy)
                    if bomb_pos not in obstacle_set and bomb_pos not in walls:
                        high_value_targets.append((bomb_pos, len(cluster)))
                        break
        
        # Находим безопасные зоны (области без бомб и препятствий)
        bombs = [tuple(b["pos"]) for b in arena.get("bombs", [])]
        bomb_set = set(bombs)
        safe_zones = []
        
        # Простой алгоритм: ищем области без препятствий и бомб
        for y in range(map_size[1]):
            for x in range(map_size[0]):
                pos = (x, y)
                if pos not in walls and pos not in obstacle_set and pos not in bomb_set:
                    # Проверяем, что рядом нет бомб
                    is_safe = True
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (x + dx, y + dy)
                        if neighbor in bomb_set:
                            is_safe = False
                            break
                    if is_safe:
                        safe_zones.append(pos)
        
        analysis = MapAnalysis(
            obstacle_density=obstacle_density,
            obstacle_clusters=clusters,
            safe_zones=safe_zones,
            high_value_targets=high_value_targets,
            map_size=map_size,
            walls=walls,
            obstacles=obstacles
        )
        
        self.map_analysis = analysis
        self.last_analysis_time = time.time()
        
        return analysis
    
    def get_map_analysis(self, force_update: bool = False) -> Optional[MapAnalysis]:
        """Получить анализ карты (кэшируется на 1 секунду)"""
        if not force_update and self.map_analysis:
            if time.time() - self.last_analysis_time < 1.0:
                return self.map_analysis
        
        try:
            data = self.get_arena()
            return self.analyze_map(data)
        except Exception:
            return self.map_analysis  # Возвращаем старый анализ при ошибке
    
    def get_boosters(self) -> Optional[Dict]:
        """Получить доступные улучшения"""
        cache_key = "boosters"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            self._rate_limit()
            response = self.session.get(f"{self.base_url}/api/booster", timeout=5)

            if response.status_code == 429:
                print(f"[WARN] Rate limited (429) for boosters, using cache")
                if cache_key in self.cache:
                    return self.cache[cache_key]
                return None  # Не критично, просто пропускаем

            if response.status_code == 400:
                # Bad Request - игра может еще не начаться или нет доступа
                print(f"[WARN] Bad Request (400) for boosters - game may not have started yet")
                return None

            # Проверяем другие ошибки перед raise_for_status
            if response.status_code >= 400:
                print(f"[WARN] HTTP {response.status_code} for boosters: {response.text}")
                return None  # Любая ошибка клиента - не критично

            response.raise_for_status()
            data = response.json()
            print(f"[BOOSTERS] Data received: {data}")
            self._set_cache(cache_key, data)
            return data
        except requests.exceptions.HTTPError as e:
            # Тихая обработка HTTP ошибок для boosters
            if e.response and e.response.status_code == 400:
                print(f"[WARN] HTTP 400 error for boosters: {e.response.text}")
                return None
            # Для других HTTP ошибок тоже возвращаем None (не критично)
            print(f"[WARN] HTTP error for boosters: {e}")
            if cache_key in self.cache:
                return self.cache[cache_key]
            return None
        except requests.exceptions.RequestException as e:
            # При любой другой ошибке возвращаем None (не критично)
            print(f"[WARN] Request error for boosters: {e}")
            if cache_key in self.cache:
                return self.cache[cache_key]
            return None
        except Exception as e:
            # Подавляем все остальные ошибки для booster (не критично)
            print(f"[WARN] Unexpected error for boosters: {e}")
            if cache_key in self.cache:
                return self.cache[cache_key]
            return None
    
    def buy_booster(self, booster_type: int) -> Dict:
        """Купить улучшение"""
        self._rate_limit()

        # Преобразуем числовой тип бустера в строковый тип API
        booster_type_mapping = {
            0: "bomb_count",      # POCKETS
            1: "bomb_range",      # BOMB_RANGE
            2: "speed",           # SPEED
            3: "vision",          # VISION
            4: "bomber_count",    # UNITS
            5: "armor",           # ARMOR
            6: "bomb_delay",      # BOMB_DELAY
            7: "passability",     # ACROBATICS
        }

        api_booster_type = booster_type_mapping.get(booster_type, str(booster_type))
        payload = {"booster": api_booster_type}

        try:
            # Сначала пробуем JSON (стандартный формат)
            response = self.session.post(f"{self.base_url}/api/booster", json=payload, timeout=5)

            if response.status_code == 400 and "invalid request format" in response.text:
                print(f"[INFO] JSON format failed for booster {booster_type} ({api_booster_type}), trying form data")
                # Если JSON не работает, пробуем form data
                response = self.session.post(f"{self.base_url}/api/booster", data=payload, timeout=5)

            response.raise_for_status()
            result = response.json()
            print(f"[SUCCESS] Bought booster {booster_type} ({api_booster_type}): {result}")

            # Инвалидируем кэш улучшений
            if "boosters" in self.cache:
                del self.cache["boosters"]

            return result

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error buying booster {booster_type} ({api_booster_type}): {e}")
            if e.response:
                print(f"   Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error buying booster {booster_type} ({api_booster_type}): {e}")
            raise
    
    def move_bombers(self, commands: List[Dict]) -> Dict:
        """Отправить команды движения"""
        if not commands:
            return {}
        
        self._rate_limit()
        payload = {"bombers": commands}
        response = self.session.post(f"{self.base_url}/api/move", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()

