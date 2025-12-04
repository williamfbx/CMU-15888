#!/home/williamfbx/15888-project/handover-sim/venv38/bin/python
"""
Policy Space Response Oracles (PSRO) for Robust Handover with GA-DDPG

Two-player zero-sum game:
- Agent (Receiver): GA-DDPG learned grasping policies
- Adversary (Perturbation): Applies perturbations to make grasping fail

Uses iterative best response training to converge to Nash equilibrium.
"""

import numpy as np
import pickle
import os
import sys
import subprocess
from datetime import datetime
from scipy.optimize import linprog
import argparse
import random
from skopt import gp_minimize
from skopt.space import Real


# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'handover-sim'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'handover-sim', 'GA-DDPG'))


class GADDPGPolicy:
    """
    GA-DDPG learned grasping policy.
    
    Each policy is a trained model checkpoint that can be evaluated.
    """
    
    def __init__(self, model_path, name=None):
        """
        Args:
            model_path: Path to trained GA-DDPG model directory (e.g., 'output/26_11_2025_16:39:00')
            name: Optional name for this policy variant
        """
        self.model_path = model_path
        self.name = name if name else os.path.basename(model_path)
    
    def get_model_path(self):
        """Get the full path to the model checkpoint"""
        return self.model_path
    
    def __repr__(self):
        return f"GADDPG({self.name})"


class PerturbationPolicy:
    """
    Adversarial perturbation policy.
    
    Applies translation and rotation perturbations to the object during handover.
    """
    
    def __init__(self, params=None, name=None):
        if params is None:
            self.params = {
                'translation_std': 0.0,
                'rotation_std': 0.0,
                'duration': 0.0,
            }
        else:
            self.params = params.copy()
        
        self.name = name if name else f"Perturb(t={params['translation_std']:.4f},r={params['rotation_std']:.4f},d={params['duration']:.2f})"
    
    def get_param_vector(self):
        """Convert params to vector"""
        return np.array([
            self.params['translation_std'],
            self.params['rotation_std'],
            self.params['duration'],
        ])
    
    @staticmethod
    def from_param_vector(vec, name=None):
        """Create policy from parameter vector"""
        params = {
            'translation_std': vec[0],
            'rotation_std': vec[1],
            'duration': vec[2],
        }
        return PerturbationPolicy(params, name)
    
    def __repr__(self):
        return self.name


class PSROTrainer:
    """
    Policy Space Response Oracles trainer for GA-DDPG vs Perturbations.
    """
    
    def __init__(self, output_dir='psro_output'):
        """
        Args:
            output_dir: Directory to save PSRO results and trained models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(output_dir, 'psro_log.txt')
        self.log_handle = open(self.log_file, 'w', buffering=1)
        print(f"Logging PSRO output to: {self.log_file}")
        
        self.agent_population = []
        self.adversary_population = []
        
        self.payoff_matrix = None
        
        self.meta_strategy_agent = None
        self.meta_strategy_adversary = None
        
        self.iteration_history = []
    
    def _log(self, message):
        """Log message to both console and file"""
        print(message)
        self.log_handle.write(message + '\n')
        self.log_handle.flush()
        
    def _initialize_populations(self, initial_agent_model=None):
        """
        Initialize populations with baseline policies.
        
        Args:
            initial_agent_model: Path to initial GA-DDPG model (if None, will train one)
        """
        self._log("=" * 80)
        self._log("Initializing populations...")
        self._log("=" * 80)
        
        # Initialize agent population with initial model
        if initial_agent_model is None:
            self._log("\nTraining initial GA-DDPG model...")
            initial_agent_model = self._train_initial_model()
        else:
            self._log(f"\nUsing provided initial GA-DDPG model: {initial_agent_model}")
        
        self.agent_population = [
            GADDPGPolicy(initial_agent_model, name="GADDPG_init")
        ]
        
        # Initialize adversary population
        adversary_variants = [
            {'translation_std': 0.0, 'rotation_std': 0.0, 'duration': 0.0},
        ]
        
        self.adversary_population.append(
            PerturbationPolicy(adversary_variants[0], name="Perturb_init")
        )
        
        # Policy population logging
        self._log(f"\nInitialized with:")
        self._log(f"  Agent population: {len(self.agent_population)} policies")
        for i, policy in enumerate(self.agent_population):
            self._log(f"    [{i}] {policy}")
        self._log(f"  Adversary population: {len(self.adversary_population)} policies")
        for i, policy in enumerate(self.adversary_population):
            self._log(f"    [{i}] {policy}")
        self._log("")
    
    def _train_initial_model(self):
        """Train initial GA-DDPG model without adversarial perturbations"""

        model_name = f"psro_init_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"        
        self._log(f"Training initial model: {model_name}")
        
        # Run training script
        psro_script = os.path.join(os.path.dirname(__file__), 'psro_train_gaddpg.py')
        cmd = [
            sys.executable,
            psro_script,
            '--model-name', model_name,
            '--no-visdom',
            '--no-test'
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to train initial model")
        
        return os.path.join('handover-sim', 'GA-DDPG', 'output', model_name)
    
    def _compute_payoff_matrix(self):
        """
        Compute empirical payoff matrix U by evaluating all (agent, adversary) pairs.
        
        U[i, j] = expected success rate when agent i plays against adversary j
        """
        self._log("=" * 80)
        self._log("Computing payoff matrix...")
        self._log("=" * 80)
        
        n_agents = len(self.agent_population)
        n_adversaries = len(self.adversary_population)
        
        self.payoff_matrix = np.zeros((n_agents, n_adversaries))
        
        for i, agent_policy in enumerate(self.agent_population):
            for j, adv_policy in enumerate(self.adversary_population):
                self._log(f"\nEvaluating: Agent[{i}] vs Adversary[{j}]")
                self._log(f"  Agent: {agent_policy}")
                self._log(f"  Adversary: {adv_policy}")
                
                success_rate = self._evaluate_policy_pair(agent_policy, adv_policy)
                self.payoff_matrix[i, j] = success_rate
                
                self._log(f"  Success rate: {success_rate:.3f}")
        
        self._log("\nPayoff Matrix (Agent rows x Adversary columns):")
        self._log(str(self.payoff_matrix))
        self._log("")
        
        return self.payoff_matrix
    
    def _evaluate_policy_pair(self, agent_policy, adv_policy, num_episodes=100):
        """
        Evaluate a single (agent, adversary) policy pair by running multiple episodes.
        
        Args:
            agent_policy: GADDPGPolicy to evaluate
            adv_policy: PerturbationPolicy to test against
            num_episodes: Number of evaluation episodes
        
        Returns:
            success_rate: Fraction of successful grasps
        """
        self._log(f"  Running {num_episodes} evaluation episodes...")
        
        # Set up paths
        ga_ddpg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'handover-sim', 'GA-DDPG'))
        model_dir = agent_policy.get_model_path()
        
        if not os.path.exists(model_dir):
            self._log(f"  WARNING: Model directory not found: {model_dir}")
            return 0.0
        
        model_name = os.path.basename(model_dir)
        test_script = os.path.abspath(os.path.join(ga_ddpg_dir, 'experiments/scripts/test_cracker_box.sh'))
        
        # Set environment variables for perturbation parameters
        env = os.environ.copy()
        env['PERTURB_TRANSLATION_STD'] = str(adv_policy.params['translation_std'])
        env['PERTURB_ROTATION_STD'] = str(adv_policy.params['rotation_std'])
        env['PERTURB_DURATION'] = str(adv_policy.params['duration'])
        
        # Run episodes one at a time and aggregate results
        total_successes = 0
        original_dir = os.getcwd()
        
        abs_output_dir = os.path.abspath(self.output_dir)
        
        try:
            os.chdir(ga_ddpg_dir)
            
            for episode_idx in range(num_episodes):
                cmd = [
                    'bash',
                    test_script,
                    model_name,
                    '1',
                    '1',
                    'latest'
                ]
                
                # Create log file
                eval_log = os.path.join(
                    abs_output_dir, 
                    f'eval_{agent_policy.name}_vs_{adv_policy.name}_ep{episode_idx}.txt'
                )
                
                with open(eval_log, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        env=env
                    )
                
                if result.returncode != 0:
                    self._log(f"    Episode {episode_idx + 1}/{num_episodes}: WARNING - failed with return code {result.returncode}")
                
                # Parse success
                success = self._parse_single_episode_success(eval_log)
                total_successes += success
                
                # Log progress every 10 episodes
                if (episode_idx + 1) % 10 == 0 or (episode_idx + 1) == num_episodes:
                    current_rate = total_successes / (episode_idx + 1)
                    self._log(f"    Progress: {episode_idx + 1}/{num_episodes} episodes, success rate: {current_rate:.3f}")
            
            # Calculate final success rate
            success_rate = total_successes / num_episodes
            self._log(f"  Final success rate: {success_rate:.3f} ({total_successes}/{num_episodes})")
            
            return success_rate
            
        except Exception as e:
            self._log(f"  ERROR during evaluation: {e}")
            return 0.0
        finally:
            os.chdir(original_dir)
    
    def _parse_single_episode_success(self, log_file):
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
            
            import re
            
            # Look for success rate in the output
            match = re.search(r'Avg\. Performance:.*?\(Success:\s*([\d.]+)', content)
            if match:
                success_rate = float(match.group(1))
                return int(success_rate > 0.5)
            
            match = re.search(r'\(Success:\s*([\d.]+)\s*\+-', content)
            if match:
                success_rate = float(match.group(1))
                return int(success_rate > 0.5)
            
            return 0
            
        except Exception as e:
            return 0
    
    def _compute_meta_nash_equilibrium(self):
        """
        Compute meta-Nash equilibrium over policy populations.
        
        This is a two-player zero-sum game where:
        - Agent (row player) wants to maximize payoff
        - Adversary (column player) wants to minimize payoff
        
        We solve for Nash equilibrium using linear programming.
        """
        self._log("=" * 80)
        self._log("Computing meta-Nash equilibrium...")
        self._log("=" * 80)
        
        n_agents = len(self.agent_population)
        n_adversaries = len(self.adversary_population)
        
        # Solve for agent's mixed strategy (maxmin problem)
        # max v subject to: U^T @ sigma_a >= v * 1, sigma_a >= 0, sum(sigma_a) = 1
        
        # Convert to minimization: min -v
        c = np.concatenate([np.zeros(n_agents), [-1]])
        
        # Inequality constraints: -U^T @ sigma_a + v * 1 <= 0
        A_ub = np.column_stack([-self.payoff_matrix.T, np.ones(n_adversaries)])
        b_ub = np.zeros(n_adversaries)
        
        # Equality constraint: sum(sigma_a) = 1
        A_eq = np.concatenate([np.ones(n_agents), [0]]).reshape(1, -1)
        b_eq = np.array([1.0])
        
        # Bounds: sigma_a >= 0, v unbounded
        bounds = [(0, None)] * n_agents + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            self._log("Warning: Failed to solve for agent strategy, using uniform distribution")
            self.meta_strategy_agent = np.ones(n_agents) / n_agents
        else:
            self.meta_strategy_agent = result.x[:n_agents]
        
        # Solve for adversary's mixed strategy (minimax problem)
        # min u subject to: U @ sigma_v <= u * 1, sigma_v >= 0, sum(sigma_v) = 1
        
        c = np.concatenate([np.zeros(n_adversaries), [1]])
        
        # Inequality constraints: U @ sigma_v - u * 1 <= 0
        A_ub = np.column_stack([self.payoff_matrix, -np.ones(n_agents)])
        b_ub = np.zeros(n_agents)
        
        # Equality constraint: sum(sigma_v) = 1
        A_eq = np.concatenate([np.ones(n_adversaries), [0]]).reshape(1, -1)
        b_eq = np.array([1.0])
        
        # Bounds: sigma_v >= 0, u unbounded
        bounds = [(0, None)] * n_adversaries + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            self._log("Warning: Failed to solve for adversary strategy, using uniform distribution")
            self.meta_strategy_adversary = np.ones(n_adversaries) / n_adversaries
        else:
            self.meta_strategy_adversary = result.x[:n_adversaries]
        
        self._log("\nMeta-Nash Equilibrium:")
        self._log(f"Agent strategy:")
        for i, prob in enumerate(self.meta_strategy_agent):
            if prob > 0.01:
                self._log(f"  [{i}] {self.agent_population[i]}: {prob:.3f}")
        
        self._log(f"\nAdversary strategy:")
        for j, prob in enumerate(self.meta_strategy_adversary):
            if prob > 0.01:
                self._log(f"  [{j}] {self.adversary_population[j]}: {prob:.3f}")
        self._log("")
        
        return self.meta_strategy_agent, self.meta_strategy_adversary
    
    def _train_best_response_agent(self, iteration, adversary_meta_strategy):
        """
        Train best response agent against adversary meta-strategy.
        
        Args:
            iteration: Current PSRO iteration number
            adversary_meta_strategy: Adversary meta-Nash strategy to train against
        
        Returns:
            new_agent_policy: Newly trained GADDPGPolicy
        """
        self._log("=" * 80)
        self._log(f"Training best response AGENT (iteration {iteration})...")
        self._log("=" * 80)
        
        # Sample adversary according to meta-strategy
        adv_idx = np.random.choice(len(self.adversary_population), p=adversary_meta_strategy)
        sampled_adversary = self.adversary_population[adv_idx]
        
        self._log(f"Training against sampled adversary: {sampled_adversary}")
        
        # Train new GA-DDPG model with adversarial perturbations
        model_name = f"psro_agent_iter{iteration}_{datetime.now().strftime('%H_%M_%S')}"
        
        # Run training with perturbations
        psro_script = os.path.join(os.path.dirname(__file__), 'psro_train_gaddpg.py')
        cmd = [
            sys.executable,
            psro_script,
            '--model-name', model_name,
            '--no-visdom',
            '--no-test',
            '--perturb-translation-std', str(sampled_adversary.params['translation_std']),
            '--perturb-rotation-std', str(sampled_adversary.params['rotation_std']),
            '--perturb-duration', str(sampled_adversary.params['duration']),
        ]
        
        self._log(f"Running training command...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to train best response agent")
        
        model_path = os.path.join('handover-sim', 'GA-DDPG', 'output', model_name)
        new_policy = GADDPGPolicy(model_path, name=f"GADDPG_iter{iteration}")
        
        self._log(f"Trained new agent: {new_policy}")
        return new_policy
    
    def _train_best_response_adversary(self, iteration, agent_meta_strategy):
        """
        Train best response adversary against agent meta-strategy.
        
        Args:
            iteration: Current PSRO iteration number
            agent_meta_strategy: Agent meta-Nash strategy to train against
        
        Returns:
            new_adversary_policy: PerturbationPolicy with optimized parameters
        """
        self._log("=" * 80)
        self._log(f"Training best response ADVERSARY (iteration {iteration})...")
        self._log("=" * 80)
        self._log(f"Optimizing perturbation parameters using Bayesian Optimization")
        
        # Define search space for perturbation parameters
        search_space = [
            Real(0.0, 2.00, name='translation_std'),
            Real(0.0, 2.00, name='rotation_std'),
            Real(0.0, 2.00, name='duration'),
        ]
        
        # Objective function: minimize expected success rate
        def objective(params):
            translation_std, rotation_std, duration = params
            
            # Create candidate adversary policy
            candidate_policy = PerturbationPolicy(
                params={
                    'translation_std': translation_std,
                    'rotation_std': rotation_std,
                    'duration': duration,
                },
                name=f"Candidate_t{translation_std:.4f}_r{rotation_std:.4f}_d{duration:.2f}"
            )
            
            self._log(f"  Evaluating: {candidate_policy}")
            
            # Evaluate against agent meta-strategy
            agent_idx = np.random.choice(len(self.agent_population), p=agent_meta_strategy)
            sampled_agent = self.agent_population[agent_idx]
            
            self._log(f"    Against sampled agent: {sampled_agent}")
            success_rate = self._evaluate_policy_pair(sampled_agent, candidate_policy)
            objective_value = -success_rate
            self._log(f"    Success rate: {success_rate:.3f}, Objective: {objective_value:.3f}")
            
            return objective_value
        
        # Run Bayesian Optimization
        self._log(f"\nRunning Bayesian Optimization...")
        
        result = gp_minimize(
            objective,
            search_space,
            n_calls=15,
            n_random_starts=5,
            random_state=iteration,
            verbose=False,
            n_jobs=1,
        )
        
        # Extract best parameters
        best_translation_std = result.x[0]
        best_rotation_std = result.x[1]
        best_duration = result.x[2]
        best_objective = result.fun
        best_success_rate = -best_objective
        
        self._log(f"\nBayesian Optimization complete!")
        self._log(f"  Best parameters found:")
        self._log(f"    translation_std: {best_translation_std:.6f}")
        self._log(f"    rotation_std: {best_rotation_std:.6f}")
        self._log(f"    duration: {best_duration:.3f}")
        self._log(f"  Expected success rate: {best_success_rate:.3f}")
        self._log(f"  Total evaluations: {len(result.func_vals)}")
        
        # Create best response adversary policy
        new_policy = PerturbationPolicy(
            params={
                'translation_std': best_translation_std,
                'rotation_std': best_rotation_std,
                'duration': best_duration,
            },
            name=f"Perturb_iter{iteration}"
        )
        
        self._log(f"\nTrained new adversary: {new_policy}")
        return new_policy

    def run_psro(self, num_iterations=5, initial_agent_model=None):
        """
        Run PSRO algorithm for specified number of iterations.
        
        Args:
            num_iterations: Number of PSRO iterations
            initial_agent_model: Path to initial GA-DDPG model checkpoint
        """
        self._log("\n" + "=" * 80)
        self._log("STARTING PSRO TRAINING")
        self._log("=" * 80 + "\n")
        
        # Initialize populations
        self._initialize_populations(initial_agent_model)
        
        # PSRO iterations
        for iteration in range(num_iterations):
            self._log("\n" + "=" * 80)
            self._log(f"PSRO ITERATION {iteration + 1}/{num_iterations}")
            self._log("=" * 80 + "\n")
            
            # Compute empirical payoff matrix
            self._compute_payoff_matrix()
            
            # Compute meta-Nash equilibrium
            self._compute_meta_nash_equilibrium()
            
            # Save current meta-strategies
            current_meta_agent = self.meta_strategy_agent.copy()
            current_meta_adversary = self.meta_strategy_adversary.copy()
            
            # Train best response agent against current adversary meta-strategy
            self._log(f"\n--- Training Agent Best Response ---")
            new_agent = self._train_best_response_agent(iteration, current_meta_adversary)
            
            # Train best response adversary against current agent meta-strategy
            self._log(f"\n--- Training Adversary Best Response ---")
            new_adversary = self._train_best_response_adversary(iteration, current_meta_agent)
            
            # Add both new policies to population
            self.agent_population.append(new_agent)
            self.adversary_population.append(new_adversary)
            
            # Save checkpoint
            self._save_checkpoint(iteration)
            
            self._log(f"\nIteration {iteration + 1} complete!")
            self._log(f"  Agent population size: {len(self.agent_population)}")
            self._log(f"  Adversary population size: {len(self.adversary_population)}")
        
        # Final evaluation with complete populations
        self._log("\n" + "=" * 80)
        self._log("FINAL EVALUATION")
        self._log("=" * 80 + "\n")
        self._compute_payoff_matrix()
        self._compute_meta_nash_equilibrium()
        
        self._log("\n" + "=" * 80)
        self._log("PSRO TRAINING COMPLETE")
        self._log("=" * 80 + "\n")
        
        self._print_final_results()
    
    def _save_checkpoint(self, iteration):
        """Save PSRO state to disk"""
        checkpoint = {
            'iteration': iteration,
            'agent_population': [
                {'model_path': p.model_path, 'name': p.name}
                for p in self.agent_population
            ],
            'adversary_population': [
                {'params': p.params, 'name': p.name}
                for p in self.adversary_population
            ],
            'payoff_matrix': self.payoff_matrix,
            'meta_strategy_agent': self.meta_strategy_agent,
            'meta_strategy_adversary': self.meta_strategy_adversary,
        }
        
        checkpoint_path = os.path.join(self.output_dir, f'psro_checkpoint_iter{iteration}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self._log(f"Saved checkpoint to: {checkpoint_path}")
        
        # Log Meta-Nash Equilibrium policies
        self._log("\nMeta-Nash Equilibrium Policies:")
        self._log("\nAgent Policies in Meta-Nash:")
        for i in range(len(self.meta_strategy_agent)):
            policy = self.agent_population[i]
            weight = self.meta_strategy_agent[i]
            full_path = os.path.abspath(policy.get_model_path())
            self._log(f"  [{i}] Weight: {weight:.3f}")
            self._log(f"      Model: {full_path}")
        
        self._log("\nAdversary Policies in Meta-Nash:")
        for j in range(len(self.meta_strategy_adversary)):
            policy = self.adversary_population[j]
            weight = self.meta_strategy_adversary[j]
            self._log(f"  [{j}] Weight: {weight:.3f}")
            self._log(f"      Params: translation_std={policy.params['translation_std']:.4f}, "
                        f"rotation_std={policy.params['rotation_std']:.4f}, "
                        f"duration={policy.params['duration']:.4f}")
    
    def _print_final_results(self):
        """Print final PSRO results"""
        self._log("Final Populations:")
        self._log("\nAgent Population:")
        for i, policy in enumerate(self.agent_population):
            prob = self.meta_strategy_agent[i]
            self._log(f"  [{i}] {policy} (weight: {prob:.3f})")
        
        self._log("\nAdversary Population:")
        for j, policy in enumerate(self.adversary_population):
            prob = self.meta_strategy_adversary[j]
            self._log(f"  [{j}] {policy} (weight: {prob:.3f})")
        
        self._log("\nFinal Payoff Matrix:")
        self._log(str(self.payoff_matrix))
        
        # Find Nash equilibrium value
        nash_value = np.dot(self.meta_strategy_agent, np.dot(self.payoff_matrix, self.meta_strategy_adversary))
        self._log(f"\nNash Equilibrium Value: {nash_value:.3f}")
        
        # Log detailed Meta-Nash Equilibrium
        self._log("\n" + "=" * 80)
        self._log("META-NASH EQUILIBRIUM DETAILS")
        self._log("=" * 80)
        
        self._log("\nAgent Policies in Meta-Nash Equilibrium:")
        for i, policy in enumerate(self.agent_population):
            weight = self.meta_strategy_agent[i]
            full_path = os.path.abspath(policy.get_model_path())
            self._log(f"\n  Policy [{i}]: {policy.name}")
            self._log(f"    Weight: {weight:.4f}")
            self._log(f"    Model Path: {full_path}")
        
        self._log("\nAdversary Policies in Meta-Nash Equilibrium:")
        for j, policy in enumerate(self.adversary_population):
            weight = self.meta_strategy_adversary[j]
            self._log(f"\n  Policy [{j}]: {policy.name}")
            self._log(f"    Weight: {weight:.4f}")
            self._log(f"    Parameters:")
            self._log(f"      - translation_std: {policy.params['translation_std']:.6f} N")
            self._log(f"      - rotation_std: {policy.params['rotation_std']:.6f} N⋅m")
            self._log(f"      - duration: {policy.params['duration']:.6f} s")
        
        self._log("\n" + "=" * 80)
        
        # Close log file
        self.log_handle.close()
        self._log = print  # Revert to print after closing


def main():
    parser = argparse.ArgumentParser(description='PSRO training for robust GA-DDPG grasping')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of PSRO iterations')
    parser.add_argument('--output-dir', type=str, 
                        default=f'psro_output/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}',
                        help='Output directory for PSRO results')
    parser.add_argument('--initial-model', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'handover-sim/GA-DDPG/output/27_11_2025_10:25:09'),
                        help='Path to initial GA-DDPG model (if None, will train one)')
    
    args = parser.parse_args()
    
    # Create PSRO trainer
    trainer = PSROTrainer(output_dir=args.output_dir)
    
    # Run PSRO
    trainer.run_psro(
        num_iterations=args.iterations,
        initial_agent_model=args.initial_model
    )


if __name__ == '__main__':
    main()