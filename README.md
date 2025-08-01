
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

Currently, we offer LLM model checkpoints that seamlessly integrate with both Gurobi and COPT optimization solver.
COPT (Cardinal Optimizer) is a mathematical optimization solver for large-scale optimization problems developed by Cardinal Operations, and it includes high-performance solvers for LP, MIP, NLP and so on.
To explore its full functionalities or to request a trial, please visit the official website: www.shanshu.ai/copt.

<img src="https://github.com/user-attachments/assets/ffd2134d-74a8-4850-8995-2dbea98d7605" style="width: 75%;">

<img src="https://github.com/user-attachments/assets/bce8e135-6587-4a89-8aac-814f6836100d" style="width: 75%;">

<img src="https://github.com/user-attachments/assets/08c23c75-ae2e-4d03-9d87-e645844bfe24" style="width: 75%;">


## Updates

- **2025.07.28** - [SIRL-Qwen2.5-7B-COPT](https://huggingface.co/chenyitian-shanshu/SIRL/tree/main/Copt) ,which leverages the COPT optimization solver, is publicly available on Hugging Face and ModelScope
- **2025.05.20** - [SIRL-Qwen2.5-7B-Gurobi](https://huggingface.co/chenyitian-shanshu/SIRL/tree/main) ,which leverages the Gurobi optimization solver, is publicly available on Hugging Face and ModelScope
- **2025.05.17** - SIRL paper published on arXiv: [Solver-Informed Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2505.11792).

## Model Release

We release the checkpoint of [SIRL-Qwen2.5-7B-Gurobi](https://huggingface.co/chenyitian-shanshu/SIRL) and [SIRL-Qwen2.5-7B-COPT](https://huggingface.co/chenyitian-shanshu/SIRL/tree/main/Copt) on Hugging Face and Model Scope. More models are coming soon.

| Solver Type          | Hugging Face      | ModelScope |
|---------------------|---------------- | ---|
| Gurobi     | [SIRL-Qwen2.5-7B-Gurobi](https://huggingface.co/chenyitian-shanshu/SIRL-Gurobi)   | [SIRL-Qwen2.5-7B-Gurobi](https://modelscope.cn/models/oneday88/SIRL-7B) |
| COPT | [SIRL-Qwen2.5-7B-COPT](https://huggingface.co/chenyitian-shanshu/SIRL-COPT) | [SIRL-Qwen2.5-7B-COPT](https://modelscope.cn/models/oneday88/sirl-qwen2-5-7b-copt) |

## Performance

We evaluated the performance of the proposed SIRL framework on four benchmarks: NL4OPT, MAMO, IndustryOR and OptMATH. 
Performance is assessed based on the pass@1 accuracy(acc). Following the rigorous evaluation protocol proposed by OptMATH, a solution is considered valid if the relative error is less than 1e-6.
The performance metrics for [SIRL](https://huggingface.co/chenyitian-shanshu/SIRL) are as follows. The highest results are highlighted in bold.

| Types         | Models            | NL4OPT | MAMO Easy | MAMO Complex | IndustryOR | OptMATH | OptiBench | Macro AVG |
|---------------|-------------------|--------|-----------|--------------|------------|---------|-----------|-----------|
| Baseline      | GPT-3.5-turbo     | 78.0%* | 79.3%*    | 33.2%*       | 21.0%*     | 15.0%*  | 47.4%*   | 51.4%*    |
|               | GPT-4             | 89.0%* | 87.3%*    | 49.3%*       | 33.3%*     | 16.6%*  | 68.6%* | 57.4%*    |
|               | Deepseek-V3       | 95.9%* | 88.3%*    | 51.1%*       | 37.0%* | 32.6%*  | **71.6%*** | 62.8%*    |
|               | DeepSeek-R1       | 82.4%  | 77.8%     | 49.3%        | **45.0%**  | 50.3% | 66.4% | 61.9% |
|               | OpenAI-O3            | 69.4%  | 70.1%     | 38.8%        | 44.0%      | 39.9% | - | 52.4% |
| Agent-based   | Chain-of-Experts  | 64.2%* | -    | -       | -     | -  | - |    -    |
|               | OptiMUS           | 78.8%* | 77.0%*    | 43.6%*       | 31.0%*     | 20.2%*   | 45.8%* | 49.4%*    |
| Offline-learning | ORLM-LLaMA-3-8B | 85.7%* | 82.3%*    | 37.4%*       | 24.0%*     | 2.6%*   | 51.1%* | 47.2%*    |
|               | LLMOpt-Qwen2.5-14B | 80.3%* | 89.5%*    | 44.1%*       | 29.0%*     | 12.5%*  | 53.8%* | 51.1%*    |
|               | OptMATH-Qwen2.5-7B | 94.7%* | 86.5%*    | 51.2%*       | 20.0%*     | 24.4%*  | 57.9%* | 55.8%*    |
| Gurobi     | SIRL-Qwen2.5-7B-Gurobi   | **96.3%**  | **90.0%**  | 62.1%     | **33.0%**   | 29.0%  | 58.0% | 61.4%     |
|               | SIRL-Qwen2.5-7B-Gurobi(pass@8) | 97.1% | 90.2% | 63.5% | 38.0% | 33.2% | 62.5% | 64.1% |
| COPT            | SIRL-Qwen2.5-7B-COPT| 95.1% | 89.3% | **68.2%** | 31.0% | **33.7%** | 58.3% | **62.6%** |
|                | SIRL-Qwen2.5-7B-COPT(pass@8) | 97.8% | 90.5% | 75.4% | 35.0% | 45.1% | 61.8% | 67.6% |

*Note:* Values marked with "*" are from original or reproduced papers with the criterion: relative error < 10⁻⁶. 

The code to reproduce the results of Gurobi version can be found in our [Jupyter Notebook for Gurobi](https://github.com/Cardinal-Operations/SIRL/blob/main/reproduce_gurobi.ipynb).
The code to reproduce the results of COPT version can be found in our [Jupyter Notebook for COPT](https://github.com/Cardinal-Operations/SIRL/blob/main/reproduce_copt.ipynb).

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
from rule_prompt_utils import gurobi_prompt_temp,copt_prompt_temp
from utils import extract_code_block, extract_obj
from vllm import SamplingParams, LLM
from langchain.prompts import PromptTemplate
import subprocess

sovler_name = 'gurobi'
# solver_name = 'copt'
# Load model and parameters for Gurobi
model = LLM("chenyitian-shanshu/SIRL-Gurobi",            
            tensor_parallel_size=1,
            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("chenyitian-shanshu/SIRL-Gurobi")
sampling_params = SamplingParams(
            n=1,
            temperature=0.5,
            top_p=0.9,
            max_tokens=8192,
            repetition_penalty=1.02
        )
# Load model and parameters for COPT
#model = LLM("chenyitian-shanshu/SIRL-COPT",            
#            tensor_parallel_size=1,
#            trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("chenyitian-shanshu/SIRL-COPT")
#sampling_params = SamplingParams(
#            n=1,
#            temperature=0.5,
#            top_p=0.9,
#            max_tokens=8192,
#            repetition_penalty=1.02
#        )
# Load question. Here is just an example. Users can replace this with datasets they want to test
question = "An industrial tire company delivers large tires for equipment to remote engineering sites either by cargo planes or ultrawide trucks. Each cargo plane can transport 10 tires per trip and costs $1000. Each ultrawide truck can transport 6 tires per trip and costs $700. The company needs to transport at least 200 tires and has available $22000. Because most remote sites don't have proper airports, the number of plane trips cannot exceed the number of ultrawide truck trips. How many trips of each should be done to minimize the total number of trips?"

# Load prompt templete for Gurobi
zeroshot_prompt_system = PromptTemplate.from_template(gurobi_prompt_temp['system'])
zeroshot_prompt_user = PromptTemplate.from_template(gurobi_prompt_temp['user'])

# Load prompt template for COPT
#zeroshot_prompt_system = PromptTemplate.from_template(copt_prompt_temp['system'])
#zeroshot_prompt_user = PromptTemplate.from_template(copt_prompt_temp['user'])
prompt =[{"role": "system", 
          "content": zeroshot_prompt_system.format().strip() }, 
         {"role": "user",
          "content": zeroshot_prompt_user.format(question=question).strip() }]   

# Generate Response
text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
response = model.generate(text,sampling_params)
response_text = response[0].outputs[0].text
code_snippet = extract_code_block(response_text,sovler_name)
result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
obj = extract_obj(result.stdout,solver_name)
print(response_text)
print('optimal value is', obj)
```

## Test Dataset
We evaluate the performance of our trained model on multiple datasets
which include NL4OPT, MAMO, IndustryOR, OptMATH. 
Minor errors exist within these testing datasets. 
To address this, we rigorously reviewed and corrected the test sets of these benchmarks, updating the questions and corresponding answers to ensure the integrity of our evaluation, with a specific focus on the NL4OPT and IndustryOR dataset. The datasets are available at [https://github.com/Cardinal-Operations/SIRL/tree/main/test_data](https://github.com/Cardinal-Operations/SIRL/tree/main/test_data). 

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
