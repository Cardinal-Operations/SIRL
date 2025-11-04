## Overview:  SIRL training framework

This folder contains the RL training scripts for the paper: **[Solver-Informed Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2505.11792)**

This part provides all the essential resources for reproducibility and custom training.
SIRL excels even when integrated with solvers that have limited prior knowledge in pre-trained models (e.g., COPT), ensuring you can seamlessly adapt our proven training methodology for your specific optimization goals.
* **Training Scripts:**  Complete scripts for model training, featuring different configurations: run_withoutKL.sh, run_withKL.sh, and run_partialKL.sh.
* **Partial-KL Implementation:** The core module containing the specific implementation of the innovative $\text{Partial-KL}$ surrogate function.
* **Reward Function:**  The specific reward function used to guide the solver in our paper.
* **Example Dataset:** A lightweight, Parquet-formatted dataset ($\approx 3000$ records) for quick testing and framework verification.

***Additional Information***
   * Our implementation builds upon the open-source codebase of [VeRL](https://github.com/volcengine/verl) .
   * All experiments for the 7B model were conducted on a single compute node equipped with eight 80GB NVIDIA H100 GPUs.

## ⚙️ Installation
### Step-1: Set up the VeRL environment
For fundamental stability and quick deployment,
we recommend using the official Docker image by following the [VeRL offical document](https://verl.readthedocs.io/en/latest/start/install.html#).
**Note**:  a local installation with the latest vllm is required for customized development
```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
# install other depend packages
pip3 install scipy pebble timeout_decorator wandb modelscope datasets  langchain coptpy gurobipy==12.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step-2: Custom logic setup (reward \& partial-KL)
Configure the domain-specific reward function and integrate the novel $\text{Partial-KL}$ surrogate function module tailored for your optimization problem.

Define the VeRL root path. Assumes you are in the project folder 'verl'; Adjust the path if 'verl' is installed outside the current directory.
```bash
VERL_ROOT="$(pwd)"
echo "Patching VeRL at: $VERL_ROOT"

# 1. Integrate custom Reward Function code
# These files provide the specialized reward signals for our optimization scheme.
cp -r SIRLTrainer/reward_func/*.py "$VERL_ROOT"/

# 2. Integrate the Partial-KL PPO implementation
# This code block overrides the standard PPO training loop with the Partial KL surrogate.
cp SIRLTrainer/partial_kl/partialKL_ppo.py "$VERL_ROOT"/verl/trainer/
cp -r SIRLTrainer/partial_kl/kl_ppo "$VERL_ROOT"/verl/trainer/
```
### Step-3: Execute training run
Initiate the main training process using the provided scripts (e.g., run_partialKL.sh) to train the LLM model.
```bash
nohup sh -x training_scripts/run_withKL 	# The baseline FullKL(reinforce++) version
nohup sh -x training_scripts/run_partialKL.sh	# The partialKL
```
---
## 📬 Contact
For any questions or issues regarding the training framework, please raise an issue on our GitHub repository or contact one of the authors via emails:
   * Yitian Chen, chenyitian@shanshu.ai
   * Minglong Cao, mlcao25@m.fudan.edu.cn
