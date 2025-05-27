
<h2 align="center"> Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling</h2>
<p align="center">
<!--     Yitian Chen<sup>*</sup>, Jingfan Xia<sup>*</sup>, Siyu Shao<sup></sup>, Dongdong Ge<sup>†</sup>, Yinyu Ye
    <br>
    <div align='center'>
        <sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Authors
    </div>
    <p align="center">
        <b>Cardinal Operations, China</b><br>
        <b>Shanghai University of Finance and Economics</b><br>
        <b>The University of Hong Kong</b><br>
        <b>Antai School of Economics and Management, Shanghai Jiao Tong University</b><br>
        <b>Department of Management Science and Engineering, Stanford University</b>
    </p> -->
    <p align="center" style="white-space: nowrap;">
        <a href="https://arxiv.org/abs/2505.11792" style="display: inline-block;"><img src='https://img.shields.io/badge/Paper-SIRL-red'></a>
        <a href="[https://huggingface.co/chenyitian-shanshu/SIRL](https://huggingface.co/chenyitian-shanshu/SIRL)" style="display: inline-block;"><img src='https://img.shields.io/badge/Model-%F0%9F%A4%97%20HuggingFace-yellow'></a>
        <a href="[https://modelscope.cn/models/oneday88/SIRL-7B](https://modelscope.cn/models/oneday88/SIRL-7B)" style="display: inline-block;"><img src="https://img.shields.io/static/v1?label=Model&message=ModeScope&color=green"></a>
    </p>
</p>


## Overview & Examples
We introduce **SIRL (Solver-Informed Reinforcement Learning)**, a novel reasoning paradigm that integrates solver feedback with reinforcement learning to train large language models (LLMs) for optimization modeling. This approach represents the first application of Reinforcement Learning with Verifiable Reward (RLVR) in the domain of optimization modeling, enabling LLMs to generate accurate mathematical formulations and code generations from natural language descriptions. SIRL leverages solver outputs to iteratively refine model performance, achieving state-of-the-art results on complex optimization tasks. The framework is particularly effective for industrial and operational research problems, where precise mathematical modeling is critical.

Particulary, we proposed surrogate function design with the Partial-KL strategy, which selectively applies the KL penalty to the mathematical formulation $\mathbf{z}^{m-1}$ and solver code $\mathbf{z}^{m}$ segments.
The Partial-KL strategy, distinct from GRPO and DAPO, effectively balances mathematical reasoning diversity with code execution rigor, showing promise for extension to tasks like AIME (math) and CodeForce (code).

<img src="https://github.com/user-attachments/assets/ffd2134d-74a8-4850-8995-2dbea98d7605" style="width: 75%;">

<img src="https://github.com/user-attachments/assets/bce8e135-6587-4a89-8aac-814f6836100d" style="width: 75%;">

<img src="https://github.com/user-attachments/assets/08c23c75-ae2e-4d03-9d87-e645844bfe24" style="width: 75%;">


## Updates

- **2025.05.17** - SIRL paper published on arXiv: [Solver-Informed Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2505.11792).

## Model Release

We release the checkpoint of [SIRL-Qwen2.5-7B](https://huggingface.co/chenyitian-shanshu/SIRL) on Hugging Face and Model Scope. More models are coming soon.

| Model Name          | Platform       |
|---------------------|----------------
| SIRL-Qwen2.5-7B     | [Hugging Face](https://huggingface.co/chenyitian-shanshu/SIRL)   | 
| SIRL-Qwen2.5-7B     | [ModelScope](https://modelscope.cn/models/oneday88/SIRL-7B)     | 

## Performance

We evaluated the performance of the proposed SIRL framework on four benchmarks: NL4OPT, MAMO, IndustryOR and OptMATH. 
Performance is assessed based on the pass@1 accuracy(acc). Following the rigorous evaluation protocol proposed by OptMATH, a solution is considered valid if the relative error is less than 1e-6.
The performance metrics for [SIRL](https://huggingface.co/chenyitian-shanshu/SIRL) are as follows. The highest results are highlighted in bold.

| Types         | Models            | NL4OPT | MAMO Easy | MAMO Complex | IndustryOR | OptMATH | Macro AVG |
|---------------|-------------------|--------|-----------|--------------|------------|---------|-----------|
| Baseline      | GPT-3.5-turbo     | 78.0%* | 79.3%*    | 33.2%*       | 21.0%*     | 15.0%*  | 45.3%*    |
|               | GPT-4             | 89.0%* | 87.3%*    | 49.3%*       | 33.0%*     | 16.6%*  | 55.0%*    |
|               | Deepseek-V3       | 95.9%* | 88.3%*    | 51.1%*       | **37.0%***     | **32.6%***  | 61.0%*    |
| Agent-based   | Chain-of-Experts  | 64.2%* | 77.2%*    | 43.6%*       | 31.0%*     | 20.2%*  | 49.4%*    |
|               | OptiMUS           | 78.8%* | 82.3%*    | 37.4%*       | 24.0%*     | 2.6%*   | 46.4%*    |
| Offline-learning | ORLM-LLaMA-3-8B | 85.7%* | 82.3%*    | 37.4%*       | 24.0%*     | 2.6%*   | 46.4%*    |
|               | LLMOpt-Qwen2.5-14B | 91.3%* | 89.5%*    | 44.1%*       | 29.0%*     | 12.5%*  | 51.1%*    |
|               | OptMATH-Qwen2.5-7B | 90.2%* | 86.5%*    | 51.2%*       | 20.0%*     | 24.4%*  | 55.4%*    |
| Online-RL     | SIRL-Qwen2.5-7B   | **96.3%**  | **90.0%**     | **62.1%**        | **33.0%**      | 29.0%   | **62.1%**     |

*Note:* Values marked with "*" are from original or reproduced papers with the criterion: relative error < 10⁻⁶. 

The code to reproduce these results can be found in our [Jupyter Notebook](https://github.com/Cardinal-Operations/SIRL/blob/main/reproduce.ipynb).

## Inference

### Setup
To get started, clone SIRL and install the required packages:

```shell
pip install -r requirements.txt
```

Make sure that you have already apply for the license of solvers such as Gurobi or COPT.

We recommend using the following prompt template which can be found in [rule_prompt_utils.py](https://github.com/Cardinal-Operations/SIRL/blob/main/rule_prompt_utils.py). Please replace the {question} with any natural language OR question.

### Quick start

Below is a simple example for model inference:

```python
from transformers import AutoTokenizer
from rule_prompt_utils import system_prompt_temp
from utils import extract_code_block, extract_obj
from vllm import SamplingParams, LLM
from langchain.prompts import PromptTemplate
import subprocess

# Load model and parameters
model = LLM("chenyitian-shanshu/SIRL",            
            tensor_parallel_size=1,
            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("chenyitian-shanshu/SIRL")
sampling_params = SamplingParams(
            n=1,
            temperature=0.5,
            top_p=0.9,
            max_tokens=8192,
            repetition_penalty=1.02
        )

# Load question. Here is just an example. Users can replace this with datasets they want to test
question = "An industrial tire company delivers large tires for equipment to remote engineering sites either by cargo planes or ultrawide trucks. Each cargo plane can transport 10 tires per trip and costs $1000. Each ultrawide truck can transport 6 tires per trip and costs $700. The company needs to transport at least 200 tires and has available $22000. Because most remote sites don't have proper airports, the number of plane trips cannot exceed the number of ultrawide truck trips. How many trips of each should be done to minimize the total number of trips?"

# Load prompt templete
zeroshot_prompt_system = PromptTemplate.from_template(system_prompt_temp['system'])
zeroshot_prompt_user = PromptTemplate.from_template(system_prompt_temp['user'])
prompt =[{"role": "system", 
          "content": zeroshot_prompt_system.format().strip() }, 
         {"role": "user",
          "content": zeroshot_prompt_user.format(question=question).strip() }]   

# Generate Response
text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
response = model.generate(text,sampling_params)
response_text = response[0].outputs[0].text
code_snippet = extract_code_block(response_text,'gurobi')
result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
obj = extract_obj(result.stdout)
print(response_text)
print('optimal value is', obj)
```

## Test Dataset
We evaluate the performance of our trained model on multiple datasets
which include NL4OPT, MAMO, IndustryOR, OptMATH. 
Minor errors exist within these testing datasets. 
To address this, we rigorously reviewed and corrected the test sets of these benchmarks, updating the questions and corresponding answers to ensure the integrity of our evaluation, with a specific focus on the NL4OPT and IndustryOR dataset. The datasets are available at [https://github.com/Cardinal-Operations/SIRL/tree/main/data/testset](https://github.com/Cardinal-Operations/SIRL/tree/main/data/testset). 

### Data Structure

Each dataset is organized in a `jsonl` file, with each line containing an independent data entry. Each entry includes:
- `en_question`: A string description of the optimization problem.
- `en_answer`: The ground truth objective function value (float). The answers of infeasible problems are "No Best Solution" or "-99999"

An example from NL4OPT:

```json
{
    "en_question": "A company needs to minimize shipping costs across 5 warehouses with varying demands...",
    "en_answer": 1250.50,
}
```



## Citation
If you find SILR useful or relevant to your research, please consider citing our paper:

```bibtex
@article{chen2025solver,
  title={Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling},
  author={Chen, Yitian and Xia, Jingfan and Shao, Siyu and Ge, Dongdong and Ye, Yinyu},
  journal={arXiv preprint arXiv:2505.11792},
  year={2025}
}
```
