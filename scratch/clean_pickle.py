import os
import pandas as pd

cache_dir = "data/iara_processed"
corrupted_count = 0
total_count = 0

print(f"Scanning directory: {cache_dir}")

for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith('.pkl'):
            total_count += 1
            path = os.path.join(root, f)
            try:
                pd.read_pickle(path)
            except Exception as e:
                # Check if it's a truncation or pickle loading error
                err_str = str(e).lower()
                if "truncated" in err_str or "unpickling" in err_str or "eof" in err_str or "ran out of input" in err_str:
                    print(f"CORRUPTED FILE FOUND: {path} - Error: {e}")
                    try:
                        os.remove(path)
                        print(f"  --> Successfully deleted: {path}")
                        corrupted_count += 1
                    except Exception as del_e:
                        print(f"  --> Failed to delete: {del_e}")

print(f"Scan complete. Total files checked: {total_count}. Corrupted files deleted: {corrupted_count}")
