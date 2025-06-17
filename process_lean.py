import json
import re
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Определяем базовую директорию проекта
BASE_DIR = Path(__file__).parent

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

def process_lean_theorems(lean_content: str) -> list[dict]:
    """
    Парсит Lean файл и извлекает утверждения теорем.
    Возвращает список словарей с информацией о теоремах.
    """
    logger.info("Processing theorems from Lean content.")
    theorems = []

    # Регулярное выражение для захвата всей декларации
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

        # Проверяем наличие решения между begin и end
        begin_end_pattern = re.compile(r'begin\s*(.*?)\s*end', re.DOTALL)
        begin_end_match = begin_end_pattern.search(full_statement)
        
        is_solved = False
        if begin_end_match:
            proof_content = begin_end_match.group(1).strip()
            # Если между begin и end есть что-то кроме пустого пространства и это не sorry
            if proof_content and 'sorry' not in proof_content:
                is_solved = True
                # Заменяем решение на sorry
                full_statement = begin_end_pattern.sub('begin\n  sorry\nend', full_statement)

        theorems.append({
            'name': name,
            'statement': full_statement,
            'is_solved': is_solved
        })

    logger.info(f"Found {len(theorems)} theorems in total.")
    return theorems

def main():
    # Используем относительные пути относительно BASE_DIR
    input_file = BASE_DIR / "miniF2F" / "lean" / "src" / "valid.lean"
    output_file = BASE_DIR / "valid.json"

    try:
        # Читаем содержимое файла
        lean_content = read_lean_file(input_file)
        
        # Обрабатываем теоремы
        theorems = process_lean_theorems(lean_content)
        
        # Сохраняем результат в JSON
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(theorems, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {len(theorems)} theorems and saved to {output_file}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 