"""
Пример использования клиента игры
"""

from game_client import GameClient

# Замените на ваш API ключ
API_KEY = "your_api_key_here"

# Для тестового сервера
TEST_SERVER = "https://games-test.datsteam.dev"

# Для основного сервера
PROD_SERVER = "https://games.datsteam.dev"

if __name__ == "__main__":
    # Создаем клиент
    client = GameClient(API_KEY, TEST_SERVER)  # Используйте TEST_SERVER для тестов
    
    # Запускаем игру
    print("Запуск клиента...")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nКлиент остановлен")

