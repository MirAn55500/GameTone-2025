"""
Веб-сервер для совместного использования данных между клиентом и визуализатором
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from game_api import GameAPI

load_dotenv()

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для визуализатора

# Инициализируем API
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL", "https://games-test.datsteam.dev")

if not api_key:
    print("⚠️  API_KEY не найден в .env файле!")
    print("Создайте файл .env и добавьте: API_KEY=your_api_key_here")
    exit(1)

game_api = GameAPI(api_key, base_url)

@app.route('/api/arena')
def get_arena():
    """Получить состояние арены (кэшируется)"""
    try:
        data = game_api.get_arena(use_cache=True)
        
        # Добавляем анализ карты
        analysis = game_api.get_map_analysis(force_update=False)
        if analysis:
            data['map_analysis'] = {
                'obstacle_density': analysis.obstacle_density,
                'cluster_count': len(analysis.obstacle_clusters),
                'safe_zones_count': len(analysis.safe_zones),
                'high_value_targets_count': len(analysis.high_value_targets),
                'high_value_targets': [
                    {'pos': pos, 'value': value} 
                    for pos, value in analysis.high_value_targets[:10]  # Топ 10 целей
                ]
            }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/map-analysis')
def get_map_analysis():
    """Получить анализ карты"""
    try:
        analysis = game_api.get_map_analysis(force_update=True)
        if not analysis:
            return jsonify({'error': 'No analysis available'}), 404
        
        return jsonify({
            'obstacle_density': analysis.obstacle_density,
            'clusters': [
                {
                    'positions': cluster,
                    'size': len(cluster),
                    'center': (
                        sum(x for x, y in cluster) // len(cluster),
                        sum(y for x, y in cluster) // len(cluster)
                    )
                }
                for cluster in analysis.obstacle_clusters
            ],
            'safe_zones': analysis.safe_zones[:50],  # Ограничиваем количество
            'high_value_targets': [
                {'pos': pos, 'value': value}
                for pos, value in analysis.high_value_targets
            ],
            'map_size': analysis.map_size,
            'total_obstacles': len(analysis.obstacles),
            'total_walls': len(analysis.walls)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/boosters')
def get_boosters():
    """Получить доступные улучшения"""
    try:
        data = game_api.get_boosters()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/buy-booster', methods=['POST'])
def buy_booster():
    """Купить улучшение"""
    try:
        data = request.json
        booster_type = data.get('booster')
        if booster_type is None:
            return jsonify({'error': 'booster type required'}), 400
        
        result = game_api.buy_booster(booster_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/move', methods=['POST'])
def move():
    """Отправить команды движения"""
    try:
        data = request.json
        commands = data.get('bombers', [])
        result = game_api.move_bombers(commands)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Главная страница - отдаем визуализатор"""
    return send_from_directory('.', 'visualizer.html')

@app.route('/<path:path>')
def serve_static(path):
    """Отдаем статические файлы"""
    return send_from_directory('.', path)

def run_game_client():
    """Запуск игрового клиента в отдельном потоке"""
    import threading
    import time
    from game_client import GameClient
    
    def client_thread():
        time.sleep(2)  # Ждем запуска сервера
        try:
            client = GameClient(api_key, base_url, use_local_api=True)
            client.run(verbose=True)
        except Exception as e:
            print(f"Ошибка клиента: {e}")
    
    thread = threading.Thread(target=client_thread, daemon=True)
    thread.start()
    return thread

def run_game_client():
    """Запуск игрового клиента в отдельном потоке"""
    import threading
    import time
    
    def client_thread():
        time.sleep(2)  # Ждем запуска сервера
        try:
            from game_client import GameClient
            print("🎮 Запуск игрового клиента...")
            client = GameClient(api_key, base_url, use_local_api=True)
            client.run(verbose=True)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Ошибка клиента: {e}")
    
    thread = threading.Thread(target=client_thread, daemon=True)
    thread.start()
    return thread

if __name__ == '__main__':
    print(f"🚀 Запуск веб-сервера и игрового клиента")
    print(f"📊 Визуализатор: http://localhost:5000/")
    print(f"🔌 API: http://localhost:5000/api/arena")
    print(f"📈 Анализ карты: http://localhost:5000/api/map-analysis")
    print(f"\n✅ Игровой клиент запускается автоматически...")
    print(f"Нажмите Ctrl+C для остановки\n")
    
    client_thread = run_game_client()
    
    # Запускаем веб-сервер
    app.run(host='0.0.0.0', port=5000, debug=False)

