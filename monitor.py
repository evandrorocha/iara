import os
import time
import sys
from datetime import datetime, timedelta

def get_default_experiment():
    test_dir = os.path.abspath("results/trainings/tests")
    if not os.path.exists(test_dir):
        return "svm_nystroem_1000_lofar"
    subdirs = []
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        if os.path.isdir(item_path) and not item.startswith("2026") and item not in ("cnn", "mlp"):
            subdirs.append((item, os.path.getmtime(item_path)))
    if subdirs:
        subdirs.sort(key=lambda x: x[1], reverse=True)
        return subdirs[0][0]
    return "svm_nystroem_1000_lofar"

# Path to the active experiment folder
if len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
else:
    EXP_NAME = get_default_experiment()

BASE_DIR = os.path.abspath(f"results/trainings/tests/{EXP_NAME}")
EVAL_DIR = os.path.join(BASE_DIR, "eval")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ANSI Terminal Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_terminal_size():
    try:
        columns, rows = os.get_terminal_size(0)
    except OSError:
        columns, rows = 80, 24
    return columns

def clear_screen():
    # Clears terminal screen cleanly
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()

def format_duration(seconds):
    if seconds is None:
        return "--:--"
    minutes, secs = divmod(int(seconds), 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def check_fold_status(fold_idx):
    fold_eval_dir = os.path.join(EVAL_DIR, f"fold_{fold_idx}")
    fold_model_dir = os.path.join(MODEL_DIR, f"fold_{fold_idx}")
    
    # Check if fold model PKL exists
    pkl_name = f"{EXP_NAME}_multiclass.pkl"
    pkl_path = os.path.join(fold_model_dir, pkl_name)
    has_model = os.path.exists(pkl_path)
    
    # Check expected CSV files
    expected_csvs = [
        f"{EXP_NAME}_multiclass_trn.csv",
        f"{EXP_NAME}_multiclass_val.csv",
        f"{EXP_NAME}_multiclass_test.csv",
        f"{EXP_NAME}_multiclass_all.csv"
    ]
    
    csv_count = 0
    if os.path.exists(fold_eval_dir):
        for csv_name in expected_csvs:
            if os.path.exists(os.path.join(fold_eval_dir, csv_name)):
                csv_count += 1
                
    # Determine status and completion timestamp
    status = "Pending"
    completed_time = None
    
    if csv_count == 4 and has_model:
        status = "Completed"
        # Use the test.csv modified time as the fold completion timestamp
        test_csv_path = os.path.join(fold_eval_dir, f"{EXP_NAME}_multiclass_test.csv")
        completed_time = os.path.getmtime(test_csv_path)
    elif os.path.exists(fold_model_dir) or os.path.exists(fold_eval_dir):
        status = "Running"
        
    return status, completed_time, csv_count, has_model

def draw_dashboard():
    clear_screen()
    cols = get_terminal_size()
    
    print("=" * cols)
    print(f"{BOLD}{BLUE}   IARA EXPERIMENT MONITOR - ACTIVE RUN{RESET}".center(cols + 10))
    print(f"{CYAN}   Experiment: {EXP_NAME}{RESET}".center(cols + 10))
    print("=" * cols)
    print()
    
    if not os.path.exists(BASE_DIR):
        print(f"{YELLOW}Waiting for the experiment output directory to be created...{RESET}")
        return False
        
    # Check all 10 folds (0 to 9)
    folds_status = []
    completed_folds = 0
    running_fold = None
    
    for i in range(10):
        status, comp_time, csvs, has_model = check_fold_status(i)
        folds_status.append({
            'fold': i,
            'status': status,
            'completed_time': comp_time,
            'csvs': csvs,
            'has_model': has_model
        })
        if status == "Completed":
            completed_folds += 1
        elif status == "Running":
            running_fold = i
            
    # Calculate timings
    durations = [None] * 10
    start_time = None
    
    # Sort completed folds by completion time to reconstruct durations
    completed_info = sorted(
        [f for f in folds_status if f['completed_time'] is not None],
        key=lambda x: x['completed_time']
    )
    
    # We can estimate the start time of the experiment from the creation of fold_0 directory
    fold0_dir = os.path.join(MODEL_DIR, "fold_0")
    if os.path.exists(fold0_dir):
        start_time = os.path.getctime(fold0_dir)
        
    if start_time and completed_info:
        # Duration of first fold is completed_time - start_time
        durations[completed_info[0]['fold']] = completed_info[0]['completed_time'] - start_time
        
        # Subsequent folds are difference between consecutive completions
        for idx in range(1, len(completed_info)):
            prev_fold = completed_info[idx-1]
            curr_fold = completed_info[idx]
            durations[curr_fold['fold']] = curr_fold['completed_time'] - prev_fold['completed_time']

    # Draw Progress Bar
    bar_width = 30
    filled_width = int(round(bar_width * completed_folds / 10.0))
    # ASCII progress bar characters compatible with all Windows terminal encodings (e.g. cp1252)
    bar = "#" * filled_width + "-" * (bar_width - filled_width)
    percentage = (completed_folds / 10.0) * 100
    
    print(f"   {BOLD}Overall Progress:  [{bar}] {percentage:.1f}% ({completed_folds}/10 folds completed){RESET}")
    print()
    
    # Table Header
    print(f"   {BOLD}{'FOLD':<8} | {'STATUS':<12} | {'DETAILS':<25} | {'DURATION':<10}{RESET}")
    print("   " + "-" * 65)
    
    now = time.time()
    for f in folds_status:
        i = f['fold']
        status = f['status']
        csvs = f['csvs']
        has_model = f['has_model']
        
        status_str = status
        color = RESET
        
        if status == "Completed":
            status_str = f"{GREEN}Completed{RESET}"
            details = f"Model saved + 4/4 CSVs"
            dur_str = format_duration(durations[i])
        elif status == "Running":
            status_str = f"{YELLOW}{BOLD}Running...{RESET}"
            color = BOLD + YELLOW
            # Details on what's done in this fold
            steps = []
            if has_model:
                steps.append("Model Saved")
            steps.append(f"{csvs}/4 CSVs Evaluated")
            details = " | ".join(steps)
            
            # Running duration estimate
            if i == 0 and start_time:
                elapsed = now - start_time
            elif i > 0 and len(completed_info) > 0:
                elapsed = now - completed_info[-1]['completed_time']
            else:
                elapsed = None
            dur_str = f"~{format_duration(elapsed)}" if elapsed else "--"
        else:
            status_str = "Pending"
            details = "Waiting..."
            dur_str = "--"
            
        print(f"   Fold {i:<3} | {status_str:<21} | {details:<25} | {dur_str:<10}")
        
    print("   " + "-" * 65)
    print()
    
    # Calculate Total Elapsed and ETA
    total_elapsed = None
    if start_time:
        total_elapsed = now - start_time
        
    avg_duration = None
    completed_durations = [d for d in durations if d is not None]
    if completed_durations:
        avg_duration = sum(completed_durations) / len(completed_durations)
        
    eta_str = "--"
    if avg_duration and completed_folds < 10:
        remaining_folds = 10 - completed_folds
        eta_seconds = remaining_folds * avg_duration
        # Subtract elapsed time of currently running fold
        if running_fold is not None:
            if running_fold == 0 and start_time:
                curr_elapsed = now - start_time
            elif running_fold > 0 and len(completed_info) > 0:
                curr_elapsed = now - completed_info[-1]['completed_time']
            else:
                curr_elapsed = 0
            eta_seconds = max(0, eta_seconds - curr_elapsed)
            
        eta_str = format_duration(eta_seconds)
        
    print(f"   {BOLD}Total Elapsed Time:{RESET} {format_duration(total_elapsed)}")
    if completed_folds < 10:
        print(f"   {BOLD}Estimated Remaining (ETA):{RESET} {GREEN}{eta_str}{RESET} (based on {format_duration(avg_duration)}/fold avg)")
    else:
        print(f"\n   *** {GREEN}{BOLD}EXPERIMENT COMPLETED SUCCESSFULLY!{RESET} ***")
        print(f"   All results are saved in {BOLD}{BASE_DIR}{RESET}")
        
    print()
    print("=" * cols)
    print(f"Auto-refreshing every 3 seconds. Press {BOLD}Ctrl+C{RESET} to exit monitor.".center(cols + 10))
    print("=" * cols)
    
    return completed_folds == 10

if __name__ == "__main__":
    try:
        while True:
            done = draw_dashboard()
            if done:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nMonitor stopped. The training continues to run in the background.")
        sys.exit(0)
