# ⚡ Optimal Bidding Strategies in ISO Energy Markets using Reinforcement Learning

**Capstone Project (Sept 2025 – Apr 2026)**  
**Queen’s University — Department of Applied Mathematics & Engineering**  
**In collaboration with Enverus (a Blackstone subsidiary)**

---

## 📘 Overview

This project develops a **Reinforcement Learning (RL)** framework to derive **optimal bidding strategies** across **Independent System Operator (ISO)** energy markets in North America.  
Rather than focusing on a single ISO (e.g., ISO-NE or PJM), the framework is designed to be **generalizable across multiple market structures**, enabling adaptive and data-driven bidding strategies that account for regional differences in price formation, demand patterns, and dispatch mechanisms.

The environment is modeled as a **Markov Decision Process (MDP)**, where each agent’s actions correspond to bid submissions and the state captures current market conditions.  
By combining **Q-Learning**, **forecasting models**, and **multi-agent game-theoretic elements**, the system aims to identify bidding policies that maximize long-term expected profit under uncertainty.

---

## 🎯 Objectives

- Develop a **Q-Learning–based bidding agent** capable of operating across multiple ISO markets.  
- Integrate **market price forecasting** using SARIMAX and related time-series models.  
- Extend to **multi-agent settings** to simulate competitive bidding dynamics.  
- Benchmark performance against baseline heuristic and rule-based strategies.  
- Build a modular and reproducible codebase compatible with industrial datasets provided by Enverus.

---

## 🧠 Technical Approach

### 1. Problem Modeling
- Market environment modeled as a **stochastic MDP**:
  - **State:** price signals, load forecasts, generation mix, time features  
  - **Action:** bid price and quantity pairs  
  - **Reward:** realized profit or surplus per dispatch interval  
- Policies π(a|s) learned to maximize discounted cumulative reward.

### 2. Learning Algorithms
- **Tabular Q-Learning** baseline for interpretability.  
- **Deep Q-Network (DQN)** extensions via `stable-baselines3` and `d3rlpy`.  
- Exploration strategies:
  - ε-greedy  
  - Boltzmann (Softmax) exploration  

### 3. Forecasting Integration
- **SARIMAX** models for short-term price prediction.  
- Forecasts incorporated as exogenous variables within the RL state representation.  

### 4. Evaluation Metrics
- Mean cumulative reward per episode  
- Profit variance and Sharpe-like ratios  
- Convergence behavior of Q-values  
- Cross-ISO generalization performance  

---

## 🧩 System Architecture - To Be Updated


---

## ⚙️ Installation & Setup

### Prerequisites
- Python ≥ 3.10  
- Virtual environment (`venv` or `conda` recommended)

### Installation
```bash
git clone https://github.com/<your-username>/capstone-optimal-bidding.git
cd capstone-optimal-bidding
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train a Q-Learning Agent
```python
python -m shell.main --mode train --n_episodes 400 --verbose --max_notional_q 0.95 --risk_penalty_lambda 0.2 --max_drawdown 50
```

---

## 📊 Run Baseline Strategies
Evaluate heuristic bidding policies to benchmark agent performance.

```python
python -m shell.main --mode baseline --baseline cost_plus --n_episodes 100
```

Available baselines include:

- `cost_plus` — bids at marginal cost plus a configurable markup  
- `quantile` — bids based on historical price quantiles  

Example with custom parameters:

```python
python -m shell.main --mode baseline --baseline quantile --quantile 0.9 --n_episodes 100
```

---

## 🧪 Run Ablation Studies
Execute the warm-start, risk constraint, and temperature schedule ablations used to validate agent design decisions.

```python
python -m shell.main --run_all_ablations --ablation_seeds 5 --ablation_episodes 50
```

Outputs include:

- Learning curve plots  
- Final reward distributions  
- CSV summaries for statistical analysis  

---

## 📉 Demand Curve Perturbation Sweep
Evaluate policy robustness under shifted demand conditions **without retraining**.

```python
python -m shell.main --plot_demand_perturbation --demand_scales 0.9,1.0,1.1 --ablation_seeds 5 --ablation_episodes 50 --eval_policy_path policy.pkl --eval_q_table_path q_table.pkl
```

This experiment measures how sensitive the learned bidding policy is to systematic load changes.

---

## 🔥 Master Ablation Run (Recommended)
Runs **all ablation studies AND the demand perturbation sweep** in a single command.

```python
python -m shell.main --run_master_ablations --ablation_seeds 5 --ablation_episodes 50 --demand_scales 0.9,1.0,1.1 --eval_policy_path policy.pkl --eval_q_table_path q_table.pkl
```

Recommended for generating the full experimental package used in reports or papers.

---

## ⚙️ Other Useful Commands

### Train with Custom Exploration
```python
python -m shell.main --mode train --temperature_mode exp_decay --temperature 1.0 --temperature_decay 0.995 --temperature_min 0.1
```

---

### Disable Warm Start
```python
python -m shell.main --mode train --warm_start_q false
```

---

### Increase Risk Sensitivity
```python
python -m shell.main --mode train --risk_penalty_lambda 0.5 --max_notional_q 0.9
```

---

### Run a Small Debug Experiment
Helpful for verifying pipeline changes quickly.

```python
python -m shell.main --mode train --n_episodes 5 --verbose
```

---

## 💡 Typical Workflow

For most experiments:

1. Train the agent  
2. Run ablations  
3. Run the demand perturbation sweep  

Example:

```python
python -m shell.main --mode train --n_episodes 400
python -m shell.main --run_master_ablations
```

## 🧑‍💻 Team Members

| Name | Role | Focus Area |
|:------|:------|:-------------|
| [**Thomas Boyle**](https://www.linkedin.com/in/thomasboyle2003/) | Developer | RL modeling, Q-Learning algorithm, SARIMAX integration |
| [**Rohan Kumar**](https://www.linkedin.com/in/rohankumar2025/) | Developer | Power Market Research, Market Environment Development |
| [**Natalie Chow**](https://www.linkedin.com/in/natalie-chow-) | Developer |  |
| [**Sam Joffe**](https://www.linkedin.com/in/samuel-joffe-ab0281308/) | Developer |  |

#### Supervisor
Dr. Serdar Yuksel
#### Industry Partner
Enverus, a Blackstone subsidiary

## 🤝 Acknowledgments

Developed as part of the **Applied Mathematics Engineering Capstone Program** at **Queen’s University**,  
in partnership with **Enverus (a Blackstone subsidiary)**.

### 🧑‍🏫 Academic Guidance
- **Dr. Serdar Yuksel**, Project Supervisor — for continuous mentorship and technical feedback.  
- The **Department of Applied Mathematics & Engineering** — for providing the academic foundation and research framework.

### 🏢 Industry Collaboration
- **Thomas Mulvihill, Adam Robinson and Dr. Juan Arteaga**, Industry Mentors — for offering domain expertise, data resources, and strategic direction.  
- **Enverus** — for partnering with Queen’s University to advance machine learning applications in energy market analytics.

---

> *This collaboration bridges theoretical research and real-world energy market optimization through reinforcement learning.*

