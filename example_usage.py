"""
Пример использования клиента игры
"""

import os
from dotenv import load_dotenv
from game_client import GameClient

# Загружаем переменные из .env
load_dotenv()

# Получаем настройки из .env
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://games.datsteam.dev")

if __name__ == "__main__":
    if not API_KEY:
        print("Ошибка: API_KEY не найден в .env файле!")
        print("Создайте файл .env и добавьте: API_KEY=your_api_key_here")
        exit(1)
    
    # Создаем клиент
    client = GameClient(API_KEY, BASE_URL)
    
    # Запускаем игру
    print("Запуск клиента...")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        client.run(verbose=True)
    except KeyboardInterrupt:
        print("\nКлиент остановлен")

