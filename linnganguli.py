import os
import re
import subprocess
import time
import json
import random
import logging
import argparse
from pathlib import Path
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log

# --- Константы и конфигурация по умолчанию ---
# Путь к исполняемому файлу Lean 4. Убедитесь, что 'lean' доступен в PATH,
# или укажите полный путь, например, '/opt/lean4/bin/lean'
LEAN_EXECUTABLE_PATH = "lean"
LEAN_VERIFICATION_TIMEOUT = 300  # Таймаут для верификации Lean в секундах (5 минут)

# API-ключ OpenRouter берется из переменной окружения
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "anthropic/claude-sonnet-4" # Модель по умолчанию

# Директории для выходных файлов и временных данных
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "log"
TEMP_DIR = BASE_DIR / "tmp"
MICRO_SUBSET_FILE = BASE_DIR / "micro_subset.txt"

# Путь к файлу valid.lean по умолчанию (относительно BASE_DIR)
VALID_LEAN_PATH_DEFAULT = BASE_DIR / "miniF2F" / "lean" / "src" / "valid.lean"

# Общее количество задач для микросабсета по умолчанию
MICRO_SUBSET_SIZE_DEFAULT = 20

# --- Настройка логирования ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "llm_requests.log"),
        logging.StreamHandler()  # Вывод также в консоль
    ]
)
logger = logging.getLogger(__name__)

def read_lean_file(filepath: Path) -> str:
    """Читает содержимое Lean файла."""
    logger.info(f"Reading Lean file: {filepath}")
    try:
        with filepath.open('r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise


def parse_lean_theorems(lean_content: str) -> list[dict]:
    """
    Парсит Lean файл и извлекает утверждения теорем.
    Возвращает список словарей: [{'name': '...', 'statement': '...', 'has_sorry': True/False}]
    Более устойчивое регулярное выражение для захвата полных деклараций.
    """
    logger.info("Parsing theorems from Lean content.")
    theorems = []

    # Регулярное выражение для захвата всей декларации (theorem, lemma, def, example и т.д.)
    # от начала строки до следующей такой же декларации или конца файла.
    theorem_blocks_pattern = re.compile(
        r'(?m)^'  # Начало строки (multiline mode)
        r'(?P<full_statement>'  # Группа 'full_statement' захватывает всю декларацию
        r'(?:theorem|lemma|def|example|instance|abbrev)\s+'  # Ключевое слово декларации
        r'(?P<name>[a-zA-Z0-9_.]+)\s*'  # Имя декларации (группа 'name')
        r'.*?)'  # Нежадный захват любого содержимого до следующей декларации или конца файла
        r'(?=\n*(?:theorem|lemma|def|example|instance|abbrev)|\Z)',  # Lookahead: до следующей декларации или конца файла
        re.DOTALL
    )

    for match in theorem_blocks_pattern.finditer(lean_content):
        full_statement = match.group('full_statement').strip()
        name = match.group('name')

        # Проверяем наличие 'sorry' во всем блоке декларации
        has_sorry = 'sorry' in full_statement

        theorems.append({
            'name': name,
            'statement': full_statement,
            'has_sorry': has_sorry
        })

    logger.info(f"Found {len(theorems)} theorems in total.")
    return theorems

def main():
    # Объявляем глобальные переменные, которые будут изменены, в самом начале функции
    global MODEL_NAME, LEAN_EXECUTABLE_PATH, LEAN_VERIFICATION_TIMEOUT
    # Также, если вы хотите изменить VALID_LEAN_PATH_DEFAULT или MICRO_SUBSET_SIZE_DEFAULT
    # global VALID_LEAN_PATH_DEFAULT, MICRO_SUBSET_SIZE_DEFAULT
    # Но лучше, чтобы эти были локальными или передавались как параметры,
    # как я и сделал ниже.

    parser = argparse.ArgumentParser(description="Automated Lean 4 theorem proving with LLM.")
    parser.add_argument("--subset_size", type=int, default=MICRO_SUBSET_SIZE_DEFAULT,
                        help="Total number of tasks for the micro-subset.")
    parser.add_argument("--lean_file", type=Path, default=VALID_LEAN_PATH_DEFAULT,
                        help="Path to the Lean file containing theorems (e.g., miniF2F/lean/src/valid.lean).")
    parser.add_argument("--model", type=str, default=MODEL_NAME, # Здесь MODEL_NAME используется как дефолтное значение, это нормально.
                        help="OpenRouter model name to use for proof generation.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (e.g., DEBUG for verbose output).")
    args = parser.parse_args()

    # Теперь присвоения будут работать корректно
    # Эти переменные не нужно объявлять global, так как они используются только в main
    MICRO_SUBSET_SIZE = args.subset_size
    VALID_LEAN_PATH = args.lean_file

    # А вот MODEL_NAME нужно, так как она используется в других функциях как глобальная
    MODEL_NAME = args.model # Изменяем глобальную переменную
    
    # Установка уровня логирования
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting MiniF2F Lean Prover Benchmark with Claude Sonnet via OpenRouter...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, Lean File='{VALID_LEAN_PATH}', Model='{MODEL_NAME}', Log Level='{args.log_level}'")
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lean_content = read_lean_file(VALID_LEAN_PATH)
    except Exception as e:
        logger.critical(f"Failed to load Lean file: {e}. Exiting.")
        return

    print(lean_content)

    try:
        all_theorems = parse_lean_theorems(lean_content)
    except Exception as e:
        logger.critical(f"Failed to parse Lean theorems: {e}. Exiting.")
        return
    
    for el in all_theorems:
        print(el)
        print(el['name'])
        print(el['statement'])
        print(el['has_sorry'])
        print("--------------------------------")


if __name__ == "__main__":
    main()