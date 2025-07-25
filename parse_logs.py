import re
from pathlib import Path
from collections import defaultdict
import argparse

# --- Configuration ---
# Set the default path to the log file.
# This can be overridden with a command-line argument.
DEFAULT_LOG_FILE = Path(__file__).parent / "logs" / "llm_requests.log"

def parse_log_file(log_path: Path) -> dict:
    """
    Parses the log file to determine the final status of each theorem.
    
    The function reads the log file line by line, looking for specific patterns 
    that indicate success (✅) or failure (❌) of a verification attempt. 
    It stores the last seen status for each theorem name.
    """
    theorem_statuses = {}
    
    # Regex to capture theorem names from success or failure log lines.
    # It looks for the emojis and extracts the theorem name that follows.
    # Example matches:
    #   - "✅ Verification successful for mathd_algebra_42." -> "mathd_algebra_42"
    #   - "❌ Verification failed for imo_1979_p1: Lean reported errors." -> "imo_1979_p1"
    #   - "❌ Failed to extract Lean proof body from LLM response for amc12a_2002_p1." -> "amc12a_2002_p1"
    #   - "❌ Timeout processing theorem some_theorem_name" -> "some_theorem_name"
    #   - "❌ Error processing theorem another_one" -> "another_one"
    status_regex = re.compile(
        r"(?:✅ Verification successful for|❌ (?:Verification failed for|Failed to extract Lean proof body from LLM response for|Timeout processing theorem|Error processing theorem))\s+([\w\d_]+)"
    )

    try:
        with log_path.open('r', encoding='utf-8') as f:
            for line in f:
                match = status_regex.search(line)
                if match:
                    theorem_name = match.group(1).strip()
                    if "✅" in line:
                        theorem_statuses[theorem_name] = "PASSED"
                    elif "❌" in line:
                        theorem_statuses[theorem_name] = "FAILED"
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")
        return {}
        
    return theorem_statuses

def main():
    """
    Main function to run the log parsing and print the summary.
    """
    parser = argparse.ArgumentParser(description="Parse benchmark logs to get final theorem statuses.")
    parser.add_argument(
        "--log_file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to the log file (default: {DEFAULT_LOG_FILE})"
    )
    args = parser.parse_args()

    print(f"Parsing log file: {args.log_file}")
    final_statuses = parse_log_file(args.log_file)
    
    if not final_statuses:
        print("No theorem statuses found in the log file.")
        return

    # --- Calculate and Print Summary ---
    total_theorems = len(final_statuses)
    passed_count = sum(1 for status in final_statuses.values() if status == "PASSED")
    failed_count = total_theorems - passed_count
    pass_rate = (passed_count / total_theorems * 100) if total_theorems > 0 else 0

    print("\n--- Final Theorem Status Summary ---")
    # Sort theorems by name for consistent output
    for theorem_name, status in sorted(final_statuses.items()):
        print(f"{theorem_name}: {status}")

    print("\n--- Overall Statistics ---")
    print(f"Total unique theorems processed: {total_theorems}")
    print(f"  - Passed: {passed_count}")
    print(f"  - Failed: {failed_count}")
    print(f"Pass Rate: {pass_rate:.2f}%")

if __name__ == "__main__":
    main() 