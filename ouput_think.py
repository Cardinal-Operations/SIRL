import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"
import warnings
warnings.filterwarnings("ignore") 
import torch
import numpy as np
from utils import load_jsonl
from vllm import LLM, SamplingParams        
from transformers import AutoTokenizer                                      
from langchain.prompts import PromptTemplate
from rule_prompt_utils import system_prompt_temp
import re
import json


# Load decode strategy
topk = 1
max_tokens = 2
repetition_penalty = 1.02 # To avoid the occasional occurrence of repeated tokens

# top-p strategy
sampling_params = SamplingParams(
    n=topk,
    temperature=0.5,
    top_p=0.9,
    max_tokens=max_tokens,
    repetition_penalty=repetition_penalty
)

file_path = "/DATA/disk1/cml/SIRL/test_data"
file_name = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP.json', 'IndustryOR_fixed.json', 'OptMATH_Bench_193.jsonl', 'OptMATH_Bench_166.jsonl','OptiBench.jsonl']
output_file_path = "/DATA/disk1/cml/SIRL/test_data_with_think"
output_file_name = ['NL4OPT_think.jsonl', 'MAMO_EasyLP_think.json', 'MAMO_ComplexLP_think.json', 'IndustryOR_fixed_think.json', 'OptMATH_Bench_193_think.jsonl', 'OptMATH_Bench_166_think.jsonl','OptiBench_think.jsonl']
# load checkpoints and tokenizer

model_path = '/DATA/disk1/cml/MScache/models/Qwen/Qwen3-32B'
tensor_parallel_size = 4

def generate_response(model,
                      text,
                      max_new_tokens=16384):
    new_output = ''
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model.generate(
                [f'{text}{new_output}'],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            new_output += outputs[0].outputs[0].text
            #print(outputs[0].outputs[0].text, end='', flush=True)
            #if new_output.endswith('</think>'):
            #    break
            match = re.search(r"</think>", new_output)
            if match:
                new_output = new_output[:match.end()]
                break

    return new_output

def main():
    print("Loading model", model_path)
    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    print("Model initialized.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# load prompt template and functions for generation
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
    
    for name,output_name in zip(file_name,output_file_name):
        file = os.path.join(file_path,name)
        output_file = os.path.join(output_file_path,output_name)
        print('Loading data', name)
        test_data = load_jsonl(file)
        print('Finish Loading')
        with open(output_file,'a',encoding = 'utf-8') as f:
            for item in test_data[:100]:
                think = generate_response(model,mp_worker(item))
                item['en_question'] = item['en_question'] + think
                json_line = json.dumps(item,ensure_ascii=False)
                f.write(json_line + "\n")
    


if __name__ == "__main__":
    main()
