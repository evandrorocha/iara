import urllib.request
import json
import base64
import os

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

def explore():
    # 1. Check signal_processing apps and acoustical
    print("=== signal_processing apps/demon.py ===")
    print(get_file("signal_processing", "apps/demon.py")[:2000])

    print("\n=== signal_processing src/lps_sp/acoustical tree ===")
    url = "https://api.github.com/repos/labsonar/signal_processing/contents/src/lps_sp/acoustical"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        for it in data:
            print(f"  [{it['type']}] {it['name']}")

    print("\n=== ml src/lps_ml/audio_processors ===")
    url = "https://api.github.com/repos/labsonar/ml/contents/src/lps_ml/audio_processors"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        for it in data:
            print(f"  [{it['type']}] {it['name']}")

    print("\n=== ml src/lps_ml/datasets ===")
    url = "https://api.github.com/repos/labsonar/ml/contents/src/lps_ml/datasets"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        for it in data:
            print(f"  [{it['type']}] {it['name']}")

    print("\n=== ml src/lps_ml/model ===")
    url = "https://api.github.com/repos/labsonar/ml/contents/src/lps_ml/model"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        for it in data:
            print(f"  [{it['type']}] {it['name']}")

if __name__ == "__main__":
    explore()
