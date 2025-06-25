import os
from openai import OpenAI
from utils import load_jsonl,extract_code_block,extract_obj,change_variable_types
from rule_prompt_utils_DS import system_prompt_temp
import subprocess
import numpy as np

client = OpenAI(api_key="sk-3e4407cc0d404e3282ed99ec348de370", base_url="https://api.deepseek.com")

input_path = '/DATA/disk1/cml/SIRL/test_data'
input_files = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP.json', 'IndustryOR_fixed.json', 'OptMATH_Bench_193.jsonl', 'OptMATH_Bench_166.jsonl','OptiBench.jsonl']

def check_result(result_str, item,solver_name='gurobi'):
    sub_answer = item['en_answer']
    # Convert sub_answer to float or None
    sub_answer = None if sub_answer == "No Best Solution" or "-9999" in str(sub_answer) else float(sub_answer)
    
    # Extract code snippet
    code_snippet = extract_code_block(result_str,solver_name)
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
                code_snippet = extract_code_block(result_str,solver_name)
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


for dataset in input_files[1:]:

    outputs = []
    snippet_package_cor = []

    data = load_jsonl(os.path.join(input_path,dataset))
    for item in data:
        response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role" : "system",
                "content" : system_prompt_temp["system"]
            },
            {
                "role" : "user",
                "content" : system_prompt_temp["user"] + item["en_question"]
            }
        ],
        stream=False
        )
        outputs.append(response.choices[0].message.content)
    for output,item in zip(outputs,data):
        snippet_package_cor.append(check_result(output, item))
    result = np.bincount(snippet_package_cor)
    #print(result)
    #print(snippet_package_cor)
    with open('pass@1_accuracy.txt','a',encoding = 'utf-8') as f:
        f.write(f'pass@1 accuracy for dataset {dataset}: {result[1]}/{sum(result)} = {result[1] / sum(result)}\n')

    """print(f'Numbers of test cases in dataset {dataset}: {sum(result)}')
    print(f'Numbers of pass@1 cases in dataset {dataset}: {result[1]}')
    print(f'pass@1 accuracy for dataset {dataset}: {result[1]}/{sum(result)} = {result[1] / sum(result)}')
    print('-------------------------------------------------------------------')"""



