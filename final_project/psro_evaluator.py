#!/home/williamfbx/15888-project/handover-sim/venv38/bin/python
"""
PSRO Nash Equilibrium Evaluator

Evaluates the Meta-Nash equilibrium strategy by sampling agents and adversaries
according to their Nash equilibrium probabilities and running evaluations.
"""

import numpy as np
import os
import sys
import subprocess
import re
import argparse
from datetime import datetime


# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'handover-sim'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'handover-sim', 'GA-DDPG'))


# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

# List of agent model paths (model directories)
NASH_AGENT_MODELS = [
    'handover-sim/GA-DDPG/output/27_11_2025_10:25:09',
]

# Meta-Nash strategy probabilities for agents
NASH_AGENT_PROBABILITIES = [
    1.0,
]

# List of perturbation policies
NASH_ADVERSARY_POLICIES = [
    {'translation_std': 0.00, 'rotation_std': 0.00, 'duration': 0.0},
]

# Meta-Nash strategy probabilities for adversaries
NASH_ADVERSARY_PROBABILITIES = [
    1.0,
]

# Number of evaluation iterations
NASH_NUM_EVAL_ITERATIONS = 100

# Output directory
EVAL_OUTPUT_DIR = f'psro_evaluation/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'

# ============================================================================

# Global log file handle
log_handle = None

def log(message):
    """Log message to both console and file"""
    print(message)
    if log_handle is not None:
        log_handle.write(message + '\n')
        log_handle.flush()


def evaluate_nash_equilibrium():
    """
    Evaluate Meta-Nash equilibrium by sampling agents and adversaries
    according to their Nash probabilities and aggregating success rates.
    
    Returns:
        average_success_rate: Mean success rate over all evaluation iterations
    """
    log("=" * 80)
    log("EVALUATING META-NASH EQUILIBRIUM")
    log("=" * 80)
    
    # Check for valid configuration
    if len(NASH_AGENT_MODELS) != len(NASH_AGENT_PROBABILITIES):
        raise ValueError(f"Agent models ({len(NASH_AGENT_MODELS)}) and probabilities ({len(NASH_AGENT_PROBABILITIES)}) must have same length")
    
    if len(NASH_ADVERSARY_POLICIES) != len(NASH_ADVERSARY_PROBABILITIES):
        raise ValueError(f"Adversary policies ({len(NASH_ADVERSARY_POLICIES)}) and probabilities ({len(NASH_ADVERSARY_PROBABILITIES)}) must have same length")
    
    if not np.isclose(sum(NASH_AGENT_PROBABILITIES), 1.0):
        raise ValueError(f"Agent probabilities must sum to 1.0, got {sum(NASH_AGENT_PROBABILITIES)}")
    
    if not np.isclose(sum(NASH_ADVERSARY_PROBABILITIES), 1.0):
        raise ValueError(f"Adversary probabilities must sum to 1.0, got {sum(NASH_ADVERSARY_PROBABILITIES)}")
    
    # Create output directory
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    
    log(f"\nConfiguration:")
    log(f"  Number of agents: {len(NASH_AGENT_MODELS)}")
    log(f"  Number of adversaries: {len(NASH_ADVERSARY_POLICIES)}")
    log(f"  Evaluation iterations: {NASH_NUM_EVAL_ITERATIONS}")
    log(f"  Output directory: {EVAL_OUTPUT_DIR}")
    
    log(f"\nAgent probabilities:")
    for i, (model, prob) in enumerate(zip(NASH_AGENT_MODELS, NASH_AGENT_PROBABILITIES)):
        log(f"  [{i}] {os.path.basename(model)}: {prob:.4f}")
    
    log(f"\nAdversary probabilities:")
    for j, (policy, prob) in enumerate(zip(NASH_ADVERSARY_POLICIES, NASH_ADVERSARY_PROBABILITIES)):
        log(f"  [{j}] t={policy['translation_std']:.4f}, r={policy['rotation_std']:.4f}, d={policy['duration']:.2f}: {prob:.4f}")
    
    # Run evaluation iterations
    log(f"\n" + "=" * 80)
    log(f"RUNNING {NASH_NUM_EVAL_ITERATIONS} EVALUATION ITERATIONS")
    log("=" * 80 + "\n")
    
    total_successes = 0
    
    for iteration in range(NASH_NUM_EVAL_ITERATIONS):
        # Sample agent according to Nash probabilities
        agent_idx = np.random.choice(len(NASH_AGENT_MODELS), p=NASH_AGENT_PROBABILITIES)
        agent_model = NASH_AGENT_MODELS[agent_idx]
        
        # Sample adversary according to Nash probabilities
        adv_idx = np.random.choice(len(NASH_ADVERSARY_POLICIES), p=NASH_ADVERSARY_PROBABILITIES)
        adv_policy = NASH_ADVERSARY_POLICIES[adv_idx]
        
        log(f"Iteration {iteration + 1}/{NASH_NUM_EVAL_ITERATIONS}:")
        log(f"  Sampled Agent[{agent_idx}]: {os.path.basename(agent_model)}")
        log(f"  Sampled Adversary[{adv_idx}]: t={adv_policy['translation_std']:.4f}, r={adv_policy['rotation_std']:.4f}, d={adv_policy['duration']:.2f}")
        
        # Evaluate pair
        success = evaluate_single_episode(agent_model, adv_policy, iteration)
        
        total_successes += success
        current_avg = total_successes / (iteration + 1)
        
        log(f"  Episode success: {success}")
        log(f"  Running average: {current_avg:.3f} ({total_successes}/{iteration + 1})")
        log("")
    
    # Final results
    average_success_rate = total_successes / NASH_NUM_EVAL_ITERATIONS
    
    log("=" * 80)
    log("EVALUATION COMPLETE")
    log("=" * 80)
    log(f"\nFinal Results:")
    log(f"  Total iterations: {NASH_NUM_EVAL_ITERATIONS}")
    log(f"  Total successes: {total_successes}")
    log(f"  Average success rate: {average_success_rate:.4f}")
    
    return average_success_rate


def evaluate_single_episode(agent_model_path, adv_policy, iteration_id):
    """
    Evaluate a single (agent, adversary) pair for 1 episode.
    
    Args:
        agent_model_path: Path to agent model directory
        adv_policy: Dictionary with perturbation parameters
        iteration_id: Iteration number for logging
    
    Returns:
        success: 1 if episode succeeded, 0 if failed
    """
    # Set up paths
    ga_ddpg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'handover-sim', 'GA-DDPG'))
    model_dir = os.path.abspath(agent_model_path)
    
    if not os.path.exists(model_dir):
        log(f"  WARNING: Model directory not found: {model_dir}")
        return 0
    
    model_name = os.path.basename(model_dir)
    
    # Use the existing test script
    test_script = os.path.abspath(os.path.join(ga_ddpg_dir, 'experiments/scripts/test_cracker_box.sh'))
    
    cmd = [
        'bash',
        test_script,
        model_name,
        '1',
        '1',
        'latest'
    ]
    
    # Set perturbation parameters
    env = os.environ.copy()
    env['PERTURB_TRANSLATION_STD'] = str(adv_policy['translation_std'])
    env['PERTURB_ROTATION_STD'] = str(adv_policy['rotation_std'])
    env['PERTURB_DURATION'] = str(adv_policy['duration'])
    
    # Run evaluation
    original_dir = os.getcwd()
    try:
        # Create log file
        eval_log = os.path.abspath(os.path.join(EVAL_OUTPUT_DIR, f'eval_iter{iteration_id}.txt'))
        
        os.chdir(ga_ddpg_dir)
        
        with open(eval_log, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env
            )
        
        if result.returncode != 0:
            log(f"  WARNING: Evaluation failed with return code {result.returncode}")
        
        # Parse success from output
        success = parse_single_episode_success(eval_log)
        
        return success
        
    except Exception as e:
        log(f"  ERROR during evaluation: {e}")
        return 0
    finally:
        os.chdir(original_dir)


def parse_single_episode_success(log_file):
    """
    Parse success/failure from single episode evaluation log.
    
    Args:
        log_file: Path to evaluation log file
    
    Returns:
        success: 1 if successful, 0 if failed
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for success rate in the output
        match = re.search(r'Avg\. Performance:.*?\(Success:\s*([\d.]+)', content)
        if match:
            success_rate = float(match.group(1))
            return int(success_rate > 0.5)
        
        match = re.search(r'\(Success:\s*([\d.]+)\s*\+-', content)
        if match:
            success_rate = float(match.group(1))
            return int(success_rate > 0.5)
        
        log(f"  WARNING: Could not parse success from {log_file}, assuming failure")
        return 0
        
    except Exception as e:
        log(f"  ERROR parsing results: {e}")
        return 0


def main():
    """Main entry point for Nash equilibrium evaluation."""
    global NASH_NUM_EVAL_ITERATIONS, log_handle
    
    parser = argparse.ArgumentParser(description='Evaluate PSRO Meta-Nash equilibrium')
    parser.add_argument('--iterations', type=int, default=None,
                        help=f'Number of evaluation iterations (default: {NASH_NUM_EVAL_ITERATIONS})')
    
    args = parser.parse_args()
    
    # Override global if specified
    if args.iterations is not None:
        NASH_NUM_EVAL_ITERATIONS = args.iterations
    
    # Setup log file
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(EVAL_OUTPUT_DIR, 'evaluation_log.txt')
    log_handle = open(log_file_path, 'w', buffering=1)
    print(f"Logging evaluation output to: {log_file_path}")
    
    try:
        # Run evaluation
        average_success_rate = evaluate_nash_equilibrium()
        
        log(f"\n Evaluation complete. Average success rate: {average_success_rate:.4f}")
        log(f" Results saved in directory: {EVAL_OUTPUT_DIR}")
    finally:
        # Close log file
        if log_handle is not None:
            log_handle.close()


if __name__ == '__main__':
    main()