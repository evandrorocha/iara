import json
import glob
import sys

# Ensure UTF-8 printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

print("=" * 80)
print("INSPECTING BASELINE MODEL CONFIGURATIONS")
print("=" * 80)

paths = glob.glob('results/trainings/tests/cnn/**/*.json', recursive=True) + \
        glob.glob('results/trainings/tests/mlp/**/*.json', recursive=True)

for path in paths:
    with open(path, encoding='utf-8') as f:
        try:
            cfg = json.load(f)
            analysis_val = None
            dataset_processor = cfg.get('dataset_processor', {})
            analysis_obj = dataset_processor.get('analysis', {})
            if 'py/tuple' in analysis_obj:
                analysis_val = analysis_obj['py/tuple'][0]
            elif 'py/reduce' in analysis_obj:
                analysis_val = analysis_obj['py/reduce'][1]['py/tuple'][0]
            else:
                analysis_val = analysis_obj
                
            input_type = cfg.get('input_type', {})
            n_windows = input_type.get('n_windows', None)
            
            print(f"File: {repr(path)}")
            print(f"  Name: {cfg.get('name')}")
            print(f"  Analysis (4=Mel, 3=Lofar): {analysis_val}")
            print(f"  n_windows (Input size): {n_windows}")
            print(f"  Output Base Dir: {cfg.get('output_base_dir')}")
            print("-" * 80)
        except Exception as e:
            print(f"Error reading {repr(path)}: {e}")
            print("-" * 80)
