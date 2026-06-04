import subprocess
import time
import sys

# Standard output configuration for utf-8
sys.stdout.reconfigure(encoding='utf-8')

commands = [
    f"docker exec iara-dev python test_scripts/svm.py -F {f} -C 4000 -A log_melgram --penalty elasticnet --l1_ratio 0.15 --reg_c 0.5"
    for f in range(1, 10)
]

print("Launching 9 folds in parallel...")
processes = []
for idx, cmd in enumerate(commands, start=1):
    print(f"Starting Fold {idx}...")
    p = subprocess.Popen(cmd, shell=True)
    processes.append(p)
    time.sleep(1.5)  # Offset starts slightly to prevent concurrent disk read spikes

print("Waiting for all processes to complete...")
for idx, p in enumerate(processes, start=1):
    p.wait()
    print(f"Fold {idx} completed with exit code {p.returncode}.")

print("All parallel training processes have completed successfully.")
