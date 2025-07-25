#!/usr/bin/env python3
"""
Script to create a small dataset with only failed tasks from the benchmark results.
"""

import json
import re
from pathlib import Path

def extract_failed_tasks_from_log(log_text):
    """Extract failed task names from the log output."""
    failed_tasks = []
    
    # Pattern to match failed tasks from the log
    pattern = r'(\w+): FAILED'
    matches = re.findall(pattern, log_text)
    
    for match in matches:
        failed_tasks.append(match)
    
    return failed_tasks

def create_failed_dataset(original_json_path, failed_tasks, output_path):
    """Create a new dataset with only the failed tasks."""
    
    # Read the original dataset
    with open(original_json_path, 'r', encoding='utf-8') as f:
        all_theorems = json.load(f)
    
    # Create a lookup dictionary for quick access
    theorem_lookup = {theorem['name']: theorem for theorem in all_theorems}
    
    # Extract only the failed tasks
    failed_theorems = []
    for task_name in failed_tasks:
        if task_name in theorem_lookup:
            failed_theorems.append(theorem_lookup[task_name])
        else:
            print(f"Warning: Task '{task_name}' not found in original dataset")
    
    # Save the failed dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(failed_theorems, f, indent=2, ensure_ascii=False)
    
    print(f"Created failed dataset with {len(failed_theorems)} tasks")
    print(f"Saved to: {output_path}")
    
    return failed_theorems

def main():
    # The log output from your benchmark
    log_output = """
2025-07-21 03:23:54,754 - INFO - amc12a_2015_p10: FAILED
2025-07-21 03:23:54,754 - INFO - induction_sum2kp1npqsqm1: FAILED
2025-07-21 03:23:54,754 - INFO - amc12a_2019_p21: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_780: FAILED
2025-07-21 03:23:54,754 - INFO - aime_1984_p5: FAILED
2025-07-21 03:23:54,754 - INFO - amc12a_2019_p9: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_13: FAILED
2025-07-21 03:23:54,754 - INFO - amc12a_2008_p4: FAILED
2025-07-21 03:23:54,754 - INFO - imo_1984_p2: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_13: FAILED
2025-07-21 03:23:54,754 - INFO - imo_2006_p6: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_221: FAILED
2025-07-21 03:23:54,754 - INFO - aime_1991_p6: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_64: FAILED
2025-07-21 03:23:54,754 - INFO - imo_1965_p1: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_33: FAILED
2025-07-21 03:23:54,754 - INFO - imo_1964_p1_2: FAILED
2025-07-21 03:23:54,754 - INFO - induction_divisibility_3divnto3m2n: PASSED
2025-07-21 03:23:54,754 - INFO - amc12_2000_p15: FAILED
2025-07-21 03:23:54,754 - INFO - imo_1987_p4: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_109: FAILED
2025-07-21 03:23:54,754 - INFO - induction_sum_1oktkp1: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_48: PASSED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_73: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_24: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_224: FAILED
2025-07-21 03:23:54,754 - INFO - amc12b_2002_p11: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_140: PASSED
2025-07-21 03:23:54,754 - INFO - algebra_amgm_prod1toneq1_sum1tongeqn: FAILED
2025-07-21 03:23:54,754 - INFO - imo_1962_p4: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_543: FAILED
2025-07-21 03:23:54,754 - INFO - algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3: FAILED
2025-07-21 03:23:54,754 - INFO - aime_1994_p4: FAILED
2025-07-21 03:23:54,754 - INFO - algebra_apb4leq8ta4pb4: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_257: FAILED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_480: PASSED
2025-07-21 03:23:54,754 - INFO - mathd_numbertheory_35: FAILED
2025-07-21 03:23:54,754 - INFO - amc12b_2002_p6: PASSED
2025-07-21 03:23:54,754 - INFO - mathd_algebra_69: FAILED
2025-07-21 03:23:54,754 - INFO - algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x: FAILED
2025-07-21 03:23:54,754 - INFO - aimeII_2020_p6: FAILED
2025-07-21 03:23:54,754 - INFO - amc12a_2010_p22: FAILED
2025-07-21 03:23:54,754 - INFO - algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_405: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1978_p5: FAILED
2025-07-21 03:23:54,755 - INFO - numbertheory_sumkmulnckeqnmul2pownm1: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2002_p21: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_155: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2017_p7: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_42: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_151: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2011_p18: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_303: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1988_p6: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1991_p1: FAILED
2025-07-21 03:23:54,755 - INFO - amc12b_2004_p3: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_59: PASSED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_709: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_461: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_43: FAILED
2025-07-21 03:23:54,755 - INFO - numbertheory_xsqpysqintdenomeq: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1973_p3: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_198: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_149: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_131: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1983_p9: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2009_p15: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_282: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_156: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2008_p15: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1974_p5: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2020_p22: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1967_p3: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1990_p2: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_232: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_530: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_77: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1988_p3: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_690: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2016_p2: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1988_p4: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2003_p1: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_405: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2003_p25: FAILED
2025-07-21 03:23:54,755 - INFO - aime_1997_p12: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_421: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_509: PASSED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_110: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2010_p10: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_31: PASSED
2025-07-21 03:23:54,755 - INFO - induction_divisibility_9div10tonm1: FAILED
2025-07-21 03:23:54,755 - INFO - aimeI_2000_p7: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1987_p6: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1966_p4: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2009_p25: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_89: FAILED
2025-07-21 03:23:54,755 - INFO - induction_divisibility_3div2tooddnp1: PASSED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_437: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1990_p3: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_326: FAILED
2025-07-21 03:23:54,755 - INFO - numbertheory_prmdvsneqnsqmodpeq0: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_126: FAILED
2025-07-21 03:23:54,755 - INFO - induction_ineq_nsqlefactn: FAILED
2025-07-21 03:23:54,755 - INFO - induction_seq_mul2pnp1: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1993_p5: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1977_p5: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2020_p13: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1964_p1_1: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1966_p5: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2003_p24: FAILED
2025-07-21 03:23:54,755 - INFO - algebra_amgm_sumasqdivbsqgeqsumbdiva: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2002_p12: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2010_p11: FAILED
2025-07-21 03:23:54,755 - INFO - amc12a_2002_p1: FAILED
2025-07-21 03:23:54,755 - INFO - amc12b_2002_p3: FAILED
2025-07-21 03:23:54,755 - INFO - imo_1979_p1: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_algebra_482: FAILED
2025-07-21 03:23:54,755 - INFO - mathd_numbertheory_668: FAILED
"""
    
    # Extract failed tasks
    failed_tasks = extract_failed_tasks_from_log(log_output)
    print(f"Found {len(failed_tasks)} failed tasks:")
    for task in failed_tasks:
        print(f"  - {task}")
    
    # Create the failed dataset
    original_json_path = Path("valid.json")
    output_path = Path("failed_tasks.json")
    
    failed_theorems = create_failed_dataset(original_json_path, failed_tasks, output_path)
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"  Total failed tasks: {len(failed_theorems)}")
    
    # Count by category
    categories = {}
    for theorem in failed_theorems:
        name = theorem['name']
        if name.startswith('amc12'):
            category = 'AMC12'
        elif name.startswith('aime'):
            category = 'AIME'
        elif name.startswith('imo'):
            category = 'IMO'
        elif name.startswith('mathd_algebra'):
            category = 'MathD Algebra'
        elif name.startswith('mathd_numbertheory'):
            category = 'MathD Number Theory'
        elif name.startswith('algebra_'):
            category = 'Algebra'
        elif name.startswith('numbertheory_'):
            category = 'Number Theory'
        elif name.startswith('induction_'):
            category = 'Induction'
        else:
            category = 'Other'
        
        categories[category] = categories.get(category, 0) + 1
    
    print(f"\nFailed tasks by category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

if __name__ == "__main__":
    main() 