import urllib.request
import json
import base64

def get_file(repo, path):
    url = f"https://api.github.com/repos/labsonar/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            if "content" in data:
                return base64.b64decode(data["content"]).decode('utf-8', errors='replace')
            return data
    except Exception as e:
        return f"Error: {e}"

# 1. Full apps/demon.py
with open("scratch/demon_full.py", "w", encoding="utf-8") as f:
    f.write(get_file("signal_processing", "apps/demon.py"))
print("Saved scratch/demon_full.py")

# 2. lps_sp/acoustical/analysis.py
with open("scratch/sp_analysis.py", "w", encoding="utf-8") as f:
    f.write(get_file("signal_processing", "src/lps_sp/acoustical/analysis.py"))
print("Saved scratch/sp_analysis.py")

# 3. lps_ml/audio_processors/spectral_processors.py
with open("scratch/spectral_processors.py", "w", encoding="utf-8") as f:
    f.write(get_file("ml", "src/lps_ml/audio_processors/spectral_processors.py"))
print("Saved scratch/spectral_processors.py")

# 4. lps_ml/audio_processors/time_processors.py
with open("scratch/time_processors.py", "w", encoding="utf-8") as f:
    f.write(get_file("ml", "src/lps_ml/audio_processors/time_processors.py"))
print("Saved scratch/time_processors.py")

# 5. lps_ml/datasets/iara.py
with open("scratch/ml_dataset_iara.py", "w", encoding="utf-8") as f:
    f.write(get_file("ml", "src/lps_ml/datasets/iara.py"))
print("Saved scratch/ml_dataset_iara.py")

# 6. Check labsonar org repositories list
url = "https://api.github.com/users/labsonar/repos"
req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
try:
    with urllib.request.urlopen(req) as resp:
        repos_data = json.loads(resp.read().decode())
        print("\nAll repos under labsonar:")
        for r in repos_data:
            print(f"  - {r['name']}: {r.get('description', '')} (Stars: {r['stargazers_count']}, Fork: {r['fork']})")
except Exception as e:
    print(f"Error listing org repos: {e}")
