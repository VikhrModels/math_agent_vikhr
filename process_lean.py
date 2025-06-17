import json
import re
from pathlib import Path
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project base directory
BASE_DIR = Path(__file__).parent

def read_lean_file(filepath: Path) -> str:
    """Reads the contents of a Lean file."""
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
    Parses Lean file and extracts theorem statements.
    Returns a list of dictionaries with theorem information.
    """
    logger.info("Processing theorems from Lean content.")
    theorems = []

    # Regular expression to capture the entire declaration
    theorem_blocks_pattern = re.compile(
        r'(?m)^'  # Start of line (multiline mode)
        r'(?P<full_statement>'  # 'full_statement' group captures the entire declaration
        r'(?:theorem|lemma|def|example|instance|abbrev)\s+'  # Declaration keyword
        r'(?P<name>[a-zA-Z0-9_.]+)\s*'  # Declaration name (group 'name')
        r'.*?)'  # Non-greedy capture of any content until next declaration or end of file
        r'(?=\n*(?:theorem|lemma|def|example|instance|abbrev)|\Z)',  # Lookahead: until next declaration or end of file
        re.DOTALL
    )

    for match in theorem_blocks_pattern.finditer(lean_content):
        full_statement = match.group('full_statement').strip()
        name = match.group('name')

        # Check for solution between begin and end
        begin_end_pattern = re.compile(r'begin\s*(.*?)\s*end', re.DOTALL)
        begin_end_match = begin_end_pattern.search(full_statement)
        
        is_solved = False
        if begin_end_match:
            proof_content = begin_end_match.group(1).strip()
            # If there is something between begin and end besides empty space and it's not sorry
            if proof_content and 'sorry' not in proof_content:
                is_solved = True
                # Replace solution with sorry
                full_statement = begin_end_pattern.sub('begin\n  sorry\nend', full_statement)

        theorems.append({
            'name': name,
            'statement': full_statement,
            'is_solved': is_solved
        })

    logger.info(f"Found {len(theorems)} theorems in total.")
    return theorems

def main():
    # Use relative paths relative to BASE_DIR
    input_file = BASE_DIR / "miniF2F" / "lean" / "src" / "valid.lean"
    output_file = BASE_DIR / "valid.json"

    try:
        # Read file contents
        lean_content = read_lean_file(input_file)
        
        # Process theorems
        theorems = process_lean_theorems(lean_content)
        
        # Save result to JSON
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(theorems, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {len(theorems)} theorems and saved to {output_file}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 