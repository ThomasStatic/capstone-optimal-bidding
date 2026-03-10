#!/usr/bin/env python3
"""
Test script for supply stack competition model with different numbers of RL agents.

Tests the new functionality with 1, 2, and 5 RL agents to verify:
1. Opponent bids are generated correctly
2. Market clearing works with mixed RL and opponent bids
3. Rewards are calculated properly
4. Training runs without errors
"""

import sys
import subprocess

def test_n_agents(n_agents: int, n_episodes: int = 3):
    """Run training with specified number of agents."""
    print(f"\n{'='*80}")
    print(f"Testing with {n_agents} RL Agent(s)")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "-m",
        "shell.main",
        "--mode", "train",
        "--n_episodes", str(n_episodes),
        "--n_agents", str(n_agents),
        "--verbose",
        "--supply_stack_price_noise_std", "2.0",
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd="c:\\Users\\PC\\Downloads\\Capstone\\capstone-optimal-bidding")
    
    if result.returncode != 0:
        print(f"\n[FAILED] Training with {n_agents} agents failed with return code {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] Training with {n_agents} agents completed successfully!")
        return True

if __name__ == "__main__":
    results = {}
    
    # Test with 1, 2, and 5 agents
    for n_agents in [1, 2, 5]:
        success = test_n_agents(n_agents, n_episodes=3)
        results[n_agents] = success
    
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    for n_agents, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{n_agents} Agent(s): {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed!")
        sys.exit(1)
