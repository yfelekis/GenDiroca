#!/usr/bin/env python3
"""
Simple runner script for modular battery optimization.
Usage examples:
    python run_battery_optimization.py --methods diroca
    python run_battery_optimization.py --methods diroca gradca
    python run_battery_optimization.py --methods all
    python run_battery_optimization.py --experiment battery --methods diroca --output-dir custom_results
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run modular battery optimization")
    parser.add_argument("--experiment", type=str, default="battery", help="Experiment name")
    parser.add_argument("--methods", nargs="+", default=["all"], 
                       choices=["diroca", "gradca", "baryca", "abslingam", "all"],
                       help="Methods to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    
    args = parser.parse_args()
    
    # Convert "all" to list of all methods
    if "all" in args.methods:
        methods = ["diroca", "gradca", "baryca", "abslingam"]
    else:
        methods = args.methods
    
    # Build command
    cmd = [sys.executable, "battery_optimization_modular.py"]
    cmd.extend(["--experiment", args.experiment])
    cmd.extend(["--methods"] + methods)
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Methods: {methods}")
    print(f"Experiment: {args.experiment}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Run the optimization
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✅ Optimization completed successfully!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Optimization failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
