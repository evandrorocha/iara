import os
import glob
import sys

# Set standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

base_dir = "results/trainings/tests"
models = [
    "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0",
    "svm_nystroem_4000_lofar_pca64_elasticnet_l1r0.15_C2.0",
    "svm_nystroem_4000_hybrid_norm_elasticnet_l1r0.15_C0.5"
]

print("Scanning for CSV files...")
for model in models:
    print(f"\nModel: {model}")
    for fold in range(10):
        # Look for standard multiclass test CSV
        std_pattern = os.path.join(base_dir, model, "eval", f"fold_{fold}", "*multiclass_test.csv")
        std_files = glob.glob(std_pattern)
        
        # Look for calibrated multiclass test CSV
        cal_pattern = os.path.join(base_dir, model, "eval", f"fold_{fold}", "*multiclass_test_b*.csv")
        cal_files = glob.glob(cal_pattern)
        
        std_status = f"Standard: {'FOUND (' + os.path.basename(std_files[0]) + ')' if std_files else 'MISSING'}"
        cal_status = f"Calibrated: {'FOUND (' + ', '.join(os.path.basename(c) for c in cal_files) + ')' if cal_files else 'MISSING'}"
        print(f"  Fold {fold}: {std_status} | {cal_status}")
