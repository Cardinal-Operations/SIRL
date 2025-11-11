# https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math_batch.py
"""
Stage-2 reward function:
Utilizing decision variables for more accurate reward capture. 
This design partially incorporates process rewards and effectively mitigates reward hacking (better than lp.file)
"""
import re
import numpy as np
import requests
import json
from collections import Counter
from executor import PythonExecutor
from content_utils import extract_code_block, extract_obj
# url = "http://10.200.250.35:8000/execute"


def code_reward(code_excu_result):
    return code_excu_result=='Done'

def answer_reward(solver_result, ans, code_excu_result, cri = 1e-6):
    abs_err = np.abs(ans) if ans else 1
    if (ans is None and solver_result is None and code_excu_result=='Done'):
        abs_err = 0
    if ans and solver_result:
        abs_err = np.abs(ans - solver_result) / (np.abs(ans) + 1)
    if ans is None:
        ans = 1
    return abs_err <cri

def sol_reward(sol_true, sol_cal):
    # Count the frequency of each element in sol_true and sol_cal
    try:
        true_counts = Counter(sol_true)
        cal_counts = Counter(sol_cal)
        
        # Calculate hit count: for each element, take min(true_counts, cal_counts)
        hit_count = 0
        hit_details = {}
        for elem in cal_counts:
            if elem in true_counts:
                matched = min(true_counts[elem], cal_counts[elem])
                hit_count += matched
                hit_details[elem] = matched
    
        true_length = len(sol_true)
        hit_rate = hit_count / true_length if true_length > 0 else 0.0
        return hit_rate
    except:
        return 0

def format_reward(processed_str: str, order:bool=False) -> bool:
    minus_score = 0

    tags = {
        'think_start':('<think>', 1),
        'think_end': ('</think>', 1),
        'model_start': ('<model>', 1),
        'model_end': ('</model>', 1),
        'python_start': ('<python>', 1),
        'python_end': ('</python>', 1)
    }

    position = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        position[tag_name] = pos = processed_str.find(tag_str)

        if count != expected_count:
            if "python" not in tag_name:
                minus_score += 1/8
            else:
                minus_score += 1/4
                
    # Verify tag order
    order_set = [
    position['think_start'], position['think_end'],
    position['model_start'], position['model_end'],
    position['python_start'], position['python_end']
]

    if order_set[1] > min(order_set[2:5]):
        minus_score += 1/3

    flag = 0
    for i in range(0, 6, 2):
        if order_set[i] > order_set[i + 1]:
            flag = 1
            break
    if flag == 1:
        minus_score += 1/3

    if order_set[4] <= max(order_set[:4]):
        minus_score += 1/3

    return 2 - minus_score

# by Batch  solution_str, (all rollout response lists)
def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    order = False
    ans_score = 1.
    sol_score = 1.
    code_score = 1.
    executor = PythonExecutor()
    response = executor.batch_apply([extract_code_block(solution_str, 'gurobi') for solution_str in solution_strs])
    obj_result =[response[0][i] for i in range(len(solution_strs))]
    code_excu_result = [response[2][i] for i in range(len(solution_strs))]
    sol_result = [response[1][i] for i in range(len(solution_strs))]
    for i in range(len(sol_result)):
        sol_result[i] = [round(x, 3) if x is not None else None for x in sol_result[i]]
    ans = [answer_reward(obj_result[i], ground_truths[i], code_excu_result[i]) for i in range(len(ground_truths))]
    if 'sol' in extra_infos[0]:
        sol = [sol_reward(extra_infos[i]['sol'], sol_result[i]) for i in range(len(ground_truths))]
    else:
        sol = [0 for i in range(len(ground_truths))]

    sol_new  = [a * b for a, b in zip(ans, sol)]

    code_ = [code_reward(code_excu_result[i]) for i in range(len(code_excu_result))]
    answers = [ans[i] * ans_score + code_[i] * code_score + sol_new[i] * sol_score for i in range(len(ans))]
    print("sample answers", answers)
    return answers
