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

# Путь к файлу valid.json по умолчанию
VALID_JSON_PATH_DEFAULT = BASE_DIR / "valid.json"

# Общее количество задач для микросабсета по умолчанию
MICRO_SUBSET_SIZE_DEFAULT = 5

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

# --- Инициализация клиента OpenRouter ---
try:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running the script.")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
except ValueError as e:
    logger.critical(e)
    exit(1) # Завершаем выполнение, если API-ключ не установлен

# --- Вспомогательные функции ---

def read_json_file(filepath: Path) -> list[dict]:
    """Читает содержимое JSON файла с теоремами."""
    logger.info(f"Reading JSON file: {filepath}")
    try:
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

def select_micro_subset_stratified(all_theorems: list[dict], subset_size: int) -> list[dict]:
    """
    Выбирает микросабсет задач, сохраняя соотношение решенных/нерешенных задач.
    """
    solved_theorems = [t for t in all_theorems if t['is_solved']]
    unsolved_theorems = [t for t in all_theorems if not t['is_solved']]

    total_solved = len(solved_theorems)
    total_unsolved = len(unsolved_theorems)
    total_all = total_solved + total_unsolved

    if total_all == 0:
        logger.warning("No theorems found to select from.")
        return []

    # Рассчитываем целевое количество задач для каждой категории
    if total_solved > 0:
        num_solved_in_subset = round(subset_size * (total_solved / total_all))
    else:
        num_solved_in_subset = 0
    
    num_unsolved_in_subset = subset_size - num_solved_in_subset

    logger.info(f"Selecting micro-subset of {subset_size} tasks:")
    logger.info(f"  Total solved theorems: {total_solved}, Target for subset: {num_solved_in_subset}")
    logger.info(f"  Total unsolved theorems: {total_unsolved}, Target for subset: {num_unsolved_in_subset}")

    # Случайный выбор из каждой категории
    selected_solved = random.sample(solved_theorems, min(num_solved_in_subset, total_solved))
    selected_unsolved = random.sample(unsolved_theorems, min(num_unsolved_in_subset, total_unsolved))
    
    micro_subset = selected_solved + selected_unsolved
    random.shuffle(micro_subset)  # Перемешиваем для случайного порядка

    # Сохраняем выбранные задачи в файл
    MICRO_SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MICRO_SUBSET_FILE.open('w', encoding='utf-8') as f:
        f.write(f"Micro-subset of {len(micro_subset)} tasks selected from {len(all_theorems)} total:\n")
        f.write(f"  Solved tasks: {len(selected_solved)}\n")
        f.write(f"  Unsolved tasks: {len(selected_unsolved)}\n\n")
        for theorem in micro_subset:
            f.write(f"- {theorem['name']} (Is solved: {theorem['is_solved']})\n")
    logger.info(f"Micro-subset details saved to {MICRO_SUBSET_FILE}")
    
    return micro_subset

def verify_lean_proof(theorem_statement: str, generated_proof_body: str) -> bool:
    """
    Создает временный Lean файл с сгенерированным доказательством
    и проверяет его с помощью Lean.
    """
    logger.info(f"Attempting Lean verification.")

    # Создаем полную теорему с сгенерированным доказательством
    full_lean_code = theorem_statement.replace("sorry", generated_proof_body, 1)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_file_path = TEMP_DIR / f"temp_proof_{time.time_ns()}.lean"

    # Добавляем необходимые импорты для успешной компиляции Lean файла
    lean_imports = """
import minif2f_import

open_locale big_operators
open_locale real
open_locale nat
open_locale topological_space
open_locale rat
"""
    
    try:
        with temp_file_path.open("w", encoding='utf-8') as f:
            f.write(lean_imports)
            f.write(full_lean_code)

        process = subprocess.run(
            [LEAN_EXECUTABLE_PATH, str(temp_file_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=LEAN_VERIFICATION_TIMEOUT
        )

        output = process.stdout + process.stderr
        
        logger.debug(f"Lean output for {temp_file_path.name}:\n{output}")

        if "error:" in output.lower():
            logger.warning(f"  ❌ Verification failed for {theorem_statement.splitlines()[0]}: Lean reported errors.")
            logger.debug(f"  Lean errors: {output}")
            return False
        
        if "warning: declaration" in output.lower() and "uses sorry" in output.lower():
            logger.warning(f"  ❌ Verification failed for {theorem_statement.splitlines()[0]}: Generated proof still uses 'sorry'.")
            return False

        logger.info(f"  ✅ Verification successful for {theorem_statement.splitlines()[0]}.")
        return True

    except FileNotFoundError:
        logger.error(f"  ❌ Lean executable '{LEAN_EXECUTABLE_PATH}' not found. Make sure Lean 4 is installed and in your PATH, or specify the full path.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ❌ Lean verification timed out for {temp_file_path.name} after {LEAN_VERIFICATION_TIMEOUT} seconds.")
        return False
    except Exception as e:
        logger.error(f"  ❌ An unexpected error occurred during Lean verification for {temp_file_path.name}: {e}")
        return False
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

def extract_proof_body(llm_response_content: str) -> str | None:
    """
    Извлекает тело доказательства из ответа LLM.
    Приоритизирует блоки кода в Markdown, затем обычные блоки begin...end.
    """
    match = re.search(r'```(?:lean)?\s*begin\s*(.*?)\s*end\s*```', llm_response_content, re.DOTALL | re.IGNORECASE)
    if match:
        logger.info("  Extracted proof body from Lean markdown block.")
        return match.group(1).strip()

    match = re.search(r'begin\s*(.*?)\s*end', llm_response_content, re.DOTALL | re.IGNORECASE)
    if match:
        logger.info("  Extracted proof body from plain begin...end block.")
        return match.group(1).strip()

    logger.warning("  Failed to extract a valid Lean proof body from LLM response.")
    return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def _call_llm_with_retry(messages: list[dict], model_name: str, extra_headers: dict, max_tokens: int, temperature: float):
    """Вспомогательная функция для вызова LLM с логикой повторных попыток."""
    return client.chat.completions.create(
        extra_headers=extra_headers,
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def generate_and_verify_proof(theorem: dict) -> bool:
    """
    Отправляет запрос к LLM для генерации доказательства и проверяет его.
    """
    logger.info(f"\n--- Processing Theorem: {theorem['name']} ---")
    logger.info(f"Original statement (first 200 chars):\n{theorem['statement'][:200]}...")

    try:
        system_prompt = (
            "Ты - высококвалифицированный Lean 4 доказыватель теорем. "
            "Твоя задача - генерировать валидные, формальные доказательства для математических теорем в Lean 4. "
            "Ты должен предоставить только Lean-код, который полностью доказывает теорему, "
            "заполняя блок `begin ... end` без использования `sorry`. "
            "Твой ответ должен содержать только Lean-код внутри блока `begin ... end` "
            "(предпочтительно обернутый в markdown-блок с указанием языка `lean`, например, ```lean ... ```)."
            "Не включай никаких дополнительных объяснений, комментариев, введения или заключения."
        )

        user_prompt = (
            f"Докажи следующую теорему на Lean 4. Заполни блок `begin ... end` без использования `sorry`. "
            f"Убедись, что доказательство является полным и корректным Lean-кодом.\n\n"
            f"```lean\n{theorem['statement']}\n```\n\n"
            f"Твой ответ должен быть только содержимым блока `begin ... end`,"
            f"обернутым в Lean-кодблок (```lean ... ```)."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.info(f"Sending request to LLM for {theorem['name']}. Prompt (first 500 chars):\n{json.dumps(messages, indent=2)[:500]}...")

        completion = _call_llm_with_retry(
            messages=messages,
            model_name=MODEL_NAME,
            max_tokens=2048,
            temperature=0.1,
        )

        generated_content = completion.choices[0].message.content
        
        logger.info(f"Received LLM response for {theorem['name']}. Content (first 200 chars):\n{generated_content[:200]}...")

        generated_proof_body = extract_proof_body(generated_content)
        if generated_proof_body:
            logger.info(f"  Attempting to verify proof for {theorem['name']}...")
            return verify_lean_proof(theorem['statement'], generated_proof_body)
        else:
            logger.warning(f"  ❌ Failed to extract Lean proof body from LLM response for {theorem['name']}. Skipping verification.")
            return False

    except Exception as e:
        logger.error(f"  An error occurred during LLM interaction for {theorem['name']}: {e}")
        return False

# --- Основная логика ---
def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser(description="Automated Lean 4 theorem proving with LLM.")
    parser.add_argument("--subset_size", type=int, default=MICRO_SUBSET_SIZE_DEFAULT,
                        help="Total number of tasks for the micro-subset.")
    parser.add_argument("--json_file", type=Path, default=VALID_JSON_PATH_DEFAULT,
                        help="Path to the JSON file containing theorems.")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="OpenRouter model name to use for proof generation.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (e.g., DEBUG for verbose output).")
    args = parser.parse_args()

    MICRO_SUBSET_SIZE = args.subset_size
    VALID_JSON_PATH = args.json_file
    MODEL_NAME = args.model
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting MiniF2F Lean Prover Benchmark with Claude Sonnet via OpenRouter...")
    logger.info(f"Configuration: Subset Size={MICRO_SUBSET_SIZE}, JSON File='{VALID_JSON_PATH}', Model='{MODEL_NAME}', Log Level='{args.log_level}'")
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        all_theorems = read_json_file(VALID_JSON_PATH)
    except Exception as e:
        logger.critical(f"Failed to load JSON file: {e}. Exiting.")
        return
    
    if not all_theorems:
        logger.warning("No theorems found in the JSON file. Exiting.")
        return

    micro_subset = select_micro_subset_stratified(all_theorems, MICRO_SUBSET_SIZE)

    if not micro_subset:
        logger.warning("No tasks selected for the micro-subset. Exiting.")
        return

    logger.info("\n--- Running LLM and Lean Verification on Micro-Subset ---")
    results = {}
    for i, theorem in enumerate(micro_subset):
        logger.info(f"\nProcessing task {i+1}/{len(micro_subset)}: {theorem['name']}")
        success = generate_and_verify_proof(theorem)
        results[theorem['name']] = success
        time.sleep(1)

    logger.info("\n--- Summary of Results ---")
    solved_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")

    logger.info(f"\nTotal tasks processed: {total_count}")
    logger.info(f"Successfully proven: {solved_count}")
    
    if total_count > 0:
        logger.info(f"Pass rate: {solved_count / total_count * 100:.2f}%")
    else:
        logger.info("No tasks processed.")

if __name__ == "__main__":
    main()