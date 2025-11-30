## Libraries
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
import argparse
import os

## From other modules
from market_loads_api import ISODemandController
from load_sarimax_projections import SARIMAXLoadProjections
from linear_approximator import PRICE_COL
from market_model import MarketModel, MarketParams
from tabular_q_agent import TabularQLearningAgent
from main import make_action_space, make_state, read_lmp_data

# Load policy - not used in code
def load_policy(policy_path: str) -> Dict:
    with open(policy_path, "rb") as f:
        policy = pickle.load(f)
    return policy

# Q-learning agent
def load_agent(q_table_path: str, num_actions: Optional[int] = None) -> TabularQLearningAgent:
    ## Load Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    ## Create agent
    if num_actions is None:
        first_state_key = next(iter(q_table.keys()))
        num_actions = len(q_table[first_state_key])
    agent = TabularQLearningAgent(num_actions=num_actions)
    
    agent.Q = q_table

    ## Note: agent set for pure exploitation and no exploration
    agent.epsilon = 0.0

    ## Print statements for verification
    print(f"Loaded agent with following parameters: \n {num_actions} actions | State in Q-table: {len(agent.Q)} | Epsilon: {agent.epsilon}")

    return agent

def save_metrics(metrics: Dict, args):
    '''Saves validation metrics to a pickle file.'''
    savedate = pd.Timestamp.now().strftime("%Y%m%d_%H-%M-%S")
    if not args.use_policy and args.exploration_rate > 0.0:
        exploration_str = f"{args.exploration_rate:.2f}".replace('.', 'p')
        filepath = os.path.join(args.analysis_path, f"validation_log_with_exploration{exploration_str}_{savedate}.pkl")
    else:
        filepath = os.path.join(args.analysis_path, f"validation_log_with_exploitation_{savedate}.pkl")
    try:
        with open(filepath, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved validation metrics to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving validation metrics to {filepath}: {e}")
        return None

def load_metrics(filepath: str) -> Dict:
    '''Loads validation metrics from a pickle file.'''
    try:
        with open(filepath, "rb") as f:
            metrics = pickle.load(f)
        print(f"Loaded validation metrics from {filepath}")
        return metrics
    except Exception as e:
        print(f"Error loading validation metrics from {filepath}: {e}")
        return {}

## Main validation function
def validate(args):
    #TODO: Implement automatic file paths
    print("\nStarting validation...")
    
    exploration_rate = args.exploration_rate

    # Optionally load policy (if you want to compare)
    policy = None
    if args.use_policy:
        policy = load_policy(args.policy_path)
        print(f"Loaded policy with {len(policy)} states and greedy selection")
        exploration_rate = 0.0  # No exploration when using a loaded policy
    else:
        print(f"No policy loaded, using Q-learning agent with softmax action selection and temperature {exploration_rate}")
    # Load Q-learning agent
    agent = load_agent(args.q_table_path)

    # Create environment components - same setup as training script
    ISO = "ERCOT"
    START_DATE = "2023-02-01"
    END_DATE = "2023-03-01"

    #TODO: Remove if unecessary 
    N_PRICE_BINS = 8
    N_LOAD_BINS = 8
    N_FORECAST_BINS = 8
    N_QTY_BINS = 8
    MAX_BID_QUANTITY_MW = 50

    N_EPSIODES = 20
    MAX_STEPS_PER_EPISODE = 24 * 7  # One week

    #TODO: use Enverus API to get historic loads
    historic_load_api = ISODemandController(START_DATE, END_DATE, ISO)
    historic_loads = historic_load_api.get_market_loads()

    #TODO: Modify SARIMAX model to not train on same data as Q-Learning algorithm
    #TODO: Save SARIMAX model to avoid retraining every time
    try:
        sarimax = SARIMAXLoadProjections(historic_loads)
        forecast_df = sarimax.get_forecast_df()
    except Exception as e:
        print(f"Warning: SARIMAX model failed, continuing without it: {e}")
        forecast_df = None

    lmp_df = read_lmp_data()

    state = make_state(historic_loads, lmp_df, forecast_df)
    action_space = make_action_space(lmp_df)

    #TODO: set market parameters appropriately, these are currently arbitrary
    market_params = MarketParams(
        marginal_cost=20.0,
        price_noise_std=5.0,
        min_price=float(action_space.price_disc.edges_[0]),
        max_price=float(action_space.price_disc.edges_[-1]),
    )
    market_model = MarketModel(action_space, market_params)
    
    # Validation metrics
    all_rewards = []
    all_actions = []
    all_states = []
    daily_rewards = []

    # Single validation run (not multiple episodes)
    print(f"\nRunning validation from {START_DATE} to {END_DATE}...")

    observation = state.reset(new_episode=False)
    state_key = agent.state_to_key(observation)
    total_reward = 0.0
    daily_reward = 0.0

    max_steps = state.n_steps() - 1

    for t in range(max_steps):
        # Select action (pure exploitation, temperature=0)
        if args.use_policy and policy is not None and state_key in policy:
            # Use loaded policy
            if isinstance(policy[state_key], (int, np.integer)):
                action = policy[state_key]
            elif isinstance(policy[state_key], np.ndarray):
                # If policy stores probabilities, select action with highest probability
                action = np.argmax(policy[state_key])
        else:
            # Use agent with softmax exploration
            action = agent.select_softmax_action(state_key, temperature=exploration_rate)

        # Get current market state
        current_time = state.current_time()
        if state.raw_state_data is not None:
            raw_current_row = state.raw_state_data.loc[current_time]
            forecast_price = raw_current_row[PRICE_COL]
            
            # Clear market and get reward
            _, _, reward = market_model.clear_market_from_action(action, forecast_price)
            
            total_reward += reward
            all_rewards.append(reward)
            all_actions.append(action)
            all_states.append(state_key)
            
            # Step to next state
            next_observation, _, done, _ = state.step(action)
            
            if done:
                break
            
            # Update state
            state_key = agent.state_to_key(next_observation)
        
        # Progress update every 24 hours
        if (t + 1) % 24 == 0:
            previous_date = pd.to_datetime(START_DATE) - pd.Timedelta(days=1) + pd.Timedelta(hours=t + 1)
            daily_rewards.append(daily_reward)
            print(f"  Day {(t + 1)//24}/{max_steps//24} | Date: {previous_date} | Daily reward: {daily_reward:.2f} | Cumulative reward: {total_reward:.2f}")
            daily_reward = 0.0
        else:
            daily_reward += reward

    # Append last day's reward if not already added
    if daily_reward != 0.0: 
        daily_rewards.append(daily_reward)

    # # Print validation results
    print(f"\n{'='*60}")
    print("Cumulative Validation Results")
    print(f"{'='*60}")
    print(f"Total days: {len(all_rewards)//24}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"\n{'='*60}")
    print("Daily Validation Results")
    print(f"{'='*60}")
    print(f"Average reward per day: {np.mean(daily_rewards):.2f}")
    print(f"Std dev daily reward: {np.std(daily_rewards):.2f}")
    print(f"Max daily reward: {np.max(daily_rewards):.2f}")
    print(f"Min daily reward: {np.min(daily_rewards):.2f}")
    print(f"{'='*60}\n")
    print("Hourly Validation Results")
    print(f"{'='*60}")
    print(f"Average reward per hour: {np.mean(all_rewards):.2f}")
    print(f"Std dev hourly reward: {np.std(all_rewards):.2f}")
    print(f"Max hourly reward: {np.max(all_rewards):.2f}")
    print(f"Min hourly reward: {np.min(all_rewards):.2f}")
    print(f"{'='*60}\n")
    
    # Create Analysis directory if it doesn't exist and save metrics
    os.makedirs(args.analysis_path, exist_ok=True)

    metrics = {
        'total_reward': total_reward,
        'hourly rewards': all_rewards,
        'daily rewards': daily_rewards,
        'actions': all_actions,
        'states': all_states,
        'avg_daily_reward': np.mean(daily_rewards),
        'std_daily_reward': np.std(daily_rewards),
        'avg_hourly_reward': np.mean(all_rewards),
        'std_hourly_reward': np.std(all_rewards)
    }


    filepath = save_metrics(metrics, args)

    # Return results for further analysis
    return metrics

if __name__ == "__main__":
    ## Argument parser for CMD line execution
    parser = argparse.ArgumentParser(description="Validate a trained Q-learning agent.")

    parser.add_argument(
    '--use_policy',  action='store_true',  default=False, help='Use a loaded policy instead of the Q-table agent.'
    )
    
    parser.add_argument('--policy_path', required=False, default="policy.pkl", help='Path to the policy file (if using policy).')

    parser.add_argument('--q_table_path', required=False, default="q_table.pkl", help='Path to the Q-table file (if using Q-table agent).')

    parser.add_argument('--analysis_path', required=False, default="./Analysis", help='Path to save the validation metrics.')

    parser.add_argument('--exploration_rate', type=float, default=0.1, help='Exploration rate for validation, only utilize if not using a loaded policy.')

    args = parser.parse_args()

    _ = validate(args)
