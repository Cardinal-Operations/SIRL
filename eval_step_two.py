import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import warnings
warnings.filterwarnings("ignore") 
import subprocess
from utils_copt import load_jsonl, extract_code_block, extract_obj, change_variable_types
import numpy as np
    


# check the pass@1 accuracy for Copt
def check_result(result_str, item):
    sub_answer = item['en_answer']
    # Convert sub_answer to float or None
    sub_answer = None if sub_answer == "No Best Solution" or "-9999" in str(sub_answer) else float(sub_answer)
    
    # Extract code snippet
    code_snippet = extract_code_block(result_str)
    with open('ouput_code.jsonl','a',encoding='utf-8') as f:
        f.write(f'{code_snippet}\n')
    #print(code_snippet)
    if not code_snippet:
        return 2
    
    # Run code snippet
    try:
        result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired:
        return 1 if sub_answer is None else 0
    
    # Check if execution failed
    if result.returncode != 0:
        return 3
    
    # Extract solver result
    solver_result = extract_obj(result.stdout)
    
    # check the first time
    if solver_result is not None and sub_answer is not None and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) <= 1e-6:
        return 1
    # Handle infeasible case or numerical mismatch since we ignore the variable types error
    if 'nfeasible' in result.stdout or (solver_result is not None and sub_answer is not None and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) > 1e-6):
        # Try re-running with modified variables: we ignore the variable types error
        result_str = change_variable_types(result_str) # change the type of variables
        if result_str:
            try:
                code_snippet = extract_code_block(result_str)
                result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
                if result.returncode == 0:
                    new_result = extract_obj(result.stdout)
                    if 'nfeasible' not in result.stdout: # infeasible and Infeasible
                        if new_result is not None and sub_answer is not None and np.abs(new_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6:
                            return 1
                        if new_result == sub_answer:
                            return 1
            except subprocess.TimeoutExpired:
                print("over_time")
    
    # Handle infeasible case after retry
    if 'nfeasible' in result.stdout:
        return 1 if sub_answer is None else 0
    
    # Final comparison
    if solver_result is not None and sub_answer is not None:
        return 1 if np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6 else 0
    return 1 if solver_result == sub_answer else 0



# if you want to check pass@1 accuracy, please run this cell
# Test the checkpoint
def main():
    
    result_strs = load_jsonl('/DATA/disk1/cml/result_strs.jsonl')
    """with open('result_strs.jsonl','r',encoding='utf-8') as f:
        for line in f:
            result_strs.append(json.loads(line))"""
    filepath = '/DATA/disk1/cml/SIRL/test_data/NL4OPT.jsonl'
    data_file = load_jsonl(filepath)
    results = []
    for result_str,data in zip(result_strs,data_file):
        results.append(check_result(result_str,data))
    
    result = np.bincount(results)
    print(result)
    """print(f'Numbers of test cases in dataset {filepath}: {sum(result)}')
    print(f'Numbers of pass@1 cases in dataset {filepath}: {result[1]}')
    print(f'pass@1 accuracy for dataset {filepath}: {result[1]}/{sum(result)} = {result[1] / sum(result)}')
    print('-------------------------------------------------------------------')

"""
if __name__ == "__main__":
    main()
  