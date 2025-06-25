import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import warnings
warnings.filterwarnings("ignore") 
import subprocess
from utils_copt import load_jsonl, extract_code_block, extract_obj, change_variable_types
import numpy as np
from vllm import LLM, SamplingParams        
from transformers import AutoTokenizer                                      
from langchain.prompts import PromptTemplate
from rule_prompt_utils_copt import system_prompt_temp
import json

# Load decode strategy
topk = 1
max_tokens = 8192
repetition_penalty = 1.02 # To avoid the occasional occurrence of repeated tokens
stop_tokens = ["</s>"]
model_path = '/DATA/disk1/cml/sft_copt/Qwen3_8B_sft'
#model_path = '/DATA/disk1/cml/MScache/models/Qwen/Qwen3-8B'
#model_path = '/DATA/disk1/cml/MScache/models/Qwen/Qwen2.5-7B-Instruct'
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
output_path = '/DATA/disk1/cml/output_files'
output_files = ['output_NL4OPT.jsonl','output_MAMO_EastLP.jsonl','output_MAMO_ComplexLP.jsonl','output_Industry_fixed.jsonl','output_OptMATH_Bench_193.jsonl','output_OptMATH_Bench_166.jsonl','output_OpttBench.jsonl']

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


def generate_with_model(model, prompt, sampling_params):   
    response = model.generate(prompt, sampling_params) 
    result_text = [g.outputs[0].text for g in response]
    return result_text


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




    for filepath,file in zip(testdataset,output_files):
    
        # loading data
        print('Loading data', filepath)
        test_data = load_jsonl(os.path.join(datapath, filepath))
        print('Finish Loading')
    
        output_file = os.path.join(output_path,file)
        # generation 
    
        prompt_list = []
        for item in test_data:
            prompt_list.append(mp_worker(item))

        result_strs = generate_with_model(model, prompt_list, sampling_params)

        codes = []
        for result_str in result_strs:
            codes.append(extract_code_block(result_str))
        
        answers = []
        for item in test_data:
            answers.append(item['en_answer'])

    
        outputs = [
                {"full_str": f, "code": c, "answer": a}
                for f, c, a in zip(result_strs, codes, answers)
                ]


        with open(output_file,'w',encoding = 'utf-8') as f:
            for result,data,item in zip(result_strs,test_data,outputs):
                if check_result(result,data) == 3:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(f'code run error {json_line}\n')
                if check_result(result,data) == 2:
                    json_line = json.dumps(item,ensure_ascii=False)
                    f.write(f'code is None {json_line}\n')

        #snippet_package_cor = []
    
        # check the pass@1 accuracy
    
        #for result_str, item in zip(result_strs, test_data):
        #    snippet_package_cor.append(check_result(result_str, item))
       


if __name__ == "__main__":
    main()
  