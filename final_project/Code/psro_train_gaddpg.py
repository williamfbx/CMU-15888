#!/home/williamfbx/15888-project/handover-sim/venv38/bin/python
"""
Main training script for GA-DDPG model
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Train GA-DDPG model')
    parser.add_argument('--config', type=str, default='td3_critic_aux_policy_aux.yaml',
                        help='Config file name (default: td3_critic_aux_policy_aux.yaml)')
    parser.add_argument('--policy', type=str, default='DDPG',
                        help='Policy name (default: DDPG)')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model directory name')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model output name (default: current timestamp)')
    parser.add_argument('--no-visdom', action='store_true',
                        help='Disable Visdom visualization')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip testing after training')
    
    # Adversarial perturbation parameters
    parser.add_argument('--perturb-translation-std', type=float, default=0.0,
                        help='Translation perturbation standard deviation (N)')
    parser.add_argument('--perturb-rotation-std', type=float, default=0.0,
                        help='Rotation perturbation standard deviation (Nm)')
    parser.add_argument('--perturb-duration', type=float, default=0.0,
                        help='Duration of perturbation (s)')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ga_ddpg_dir = os.path.join(script_dir, 'handover-sim', 'GA-DDPG')
    if not os.path.exists(ga_ddpg_dir):
        print(f"Error: GA-DDPG directory not found at {ga_ddpg_dir}")
        sys.exit(1)
    
    os.chdir(ga_ddpg_dir)
    
    # Generate model name if not provided
    model_name = args.model_name
    if model_name is None:
        model_name = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = 'True'
    
    # Create output directory
    output_dir = os.path.join('output', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build training command
    python_path = sys.executable
    
    cmd = [
        python_path, '-m', 'core.train_online',
        '--save_model',
        '--config_file', args.config,
        '--policy', args.policy,
        '--log',
        '--fix_output_time', model_name
    ]
    
    if args.pretrained:
        pretrained_path = os.path.join('output', args.pretrained)
        cmd.extend(['--pretrained', pretrained_path])
    
    if not args.no_visdom:
        cmd.append('--visdom')
    
    # Add perturbation parameters
    if args.perturb_translation_std > 0 or args.perturb_rotation_std > 0:
        cmd.extend(['--perturb_translation_std', str(args.perturb_translation_std)])
        cmd.extend(['--perturb_rotation_std', str(args.perturb_rotation_std)])
        cmd.extend(['--perturb_duration', str(args.perturb_duration)])
        print(f"Training with perturbations: translation={args.perturb_translation_std}m, "
              f"rotation={args.perturb_rotation_std}rad, duration={args.perturb_duration}s")
    
    # Log file
    log_file = os.path.join(output_dir, 'log.txt')
    print(f"Logging output to {log_file}")
    
    # Run training
    print(f"Running training with command: {' '.join(cmd)}")
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\nTraining failed with exit code {process.returncode}")
            sys.exit(process.returncode)
    
    print("\nTraining completed successfully!")
    
    # Run testing
    if not args.no_test:
        print("\nRunning evaluation...")
        test_script = os.path.join('experiments', 'scripts', 'test_cracker_box.sh')
        if os.path.exists(test_script):
            test_cmd = ['bash', test_script, model_name]
            print(f"Running: {' '.join(test_cmd)}")
            subprocess.run(test_cmd, check=False)
        else:
            print(f"Test script not found: {test_script}")
    
    print(f"\nModel saved to: {output_dir}")

if __name__ == '__main__':
    main()