import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
warnings.filterwarnings("ignore") 
import subprocess
from utils_copt import load_jsonl, extract_code_block, extract_obj, change_variable_types
import numpy as np
from vllm import LLM, SamplingParams        
from transformers import AutoTokenizer                                      
from langchain.prompts import PromptTemplate
from rule_prompt_utils_copt import system_prompt_temp

# Load decode strategy
topk = 1
max_tokens = 8192
repetition_penalty = 1.02 # To avoid the occasional occurrence of repeated tokens
stop_tokens = ["</s>"]
#model_path = '/DATA/disk1/cml/sft_copt/Qwen3_8B_sft'
model_path = '/DATA/disk1/cml/MScache/models/Qwen/Qwen3-8B'
tensor_parallel_size = 1
solver_name = 'copt'

# top-p strategy
sampling_params = SamplingParams(
    n=topk,
    temperature=0.5,
    top_p=0.9,
    max_tokens=max_tokens,
    stop=stop_tokens,
    repetition_penalty=repetition_penalty
)
datapath = '/DATA/disk1/cml/SIRL/test_data'
testdataset = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP.json', 'IndustryOR_fixed.json', 'OptMATH_Bench_193.jsonl', 'OptMATH_Bench_166.jsonl','OptiBench.jsonl']

def generate_with_model(model, prompt, sampling_params):   
    response = model.generate(prompt, sampling_params) 
    result_text = [g.outputs[0].text for g in response]
    return result_text



# check the pass@1 accuracy for Copt
def check_result(result_str, item):
    sub_answer = item['en_answer']
    # Convert sub_answer to float or None
    sub_answer = None if sub_answer == "No Best Solution" or "-9999" in str(sub_answer) else float(sub_answer)
    
    # Extract code snippet
    code_snippet = extract_code_block(result_str)
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
    # load checkpoints and tokenizer

    print("Loading model", model_path)
    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    print("Model initialized.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # load prompt template and functions for Copt generation
    zeroshot_prompt_system = PromptTemplate.from_template(system_prompt_temp['system'])
    zeroshot_prompt_user = PromptTemplate.from_template(system_prompt_temp['user'])
    def mp_worker(item):
        prompt = [
            {
                "role": "system",
                "content": zeroshot_prompt_system.format(question=item['en_question']).strip()
            },
            {
                "role": "user",
                "content": zeroshot_prompt_user.format(question=item['en_question']).strip()
            }
        ]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return text




    for filepath in testdataset[6:]:
    
        # loading data
        print('Loading data', filepath)
        test_data = load_jsonl(os.path.join(datapath, filepath))
        print('Finish Loading')
    
        # generation 
    
        prompt_list = []
        for item in test_data:
            prompt_list.append(mp_worker(item))
        result_strs = generate_with_model(model, prompt_list, sampling_params)
        snippet_package_cor = []
    
        # check the pass@1 accuracy
        #index = 0
        for result_str, item in zip(result_strs, test_data):
            snippet_package_cor.append(check_result(result_str, item))
            #if check_result(result_str, item) == 3:
            #    with open('error_code.txt','a',encoding='utf-8') as f:
            #        f.write(f'the {index} error code is\n{extract_code_block(result_str)}\n')
                    #f.write(f'the {index} error code is\n{result_str}\n')
            #index += 1
            with open('code.txt','a',encoding = 'utf-8') as f:
                f.write(f'{extract_code_block(result_str)}\n')
        result = np.bincount(snippet_package_cor)
        #print(result)
        print(f'Numbers of test cases in dataset {filepath}: {sum(result)}')
        print(f'Numbers of pass@1 cases in dataset {filepath}: {result[1]}')
        print(f'pass@1 accuracy for dataset {filepath}: {result[1]}/{sum(result)} = {result[1] / sum(result)}')
        print('-------------------------------------------------------------------')


if __name__ == "__main__":
    main()
  