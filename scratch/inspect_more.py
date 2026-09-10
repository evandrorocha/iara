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

# 1. template_project Libraries.txt and manager
print("=== template_project Libraries.txt ===")
print(get_file("template_project", "Libraries.txt"))

print("=== template_project README.md ===")
print(get_file("template_project", "README.md")[:1000])

# 2. signal_processing requirements.txt and README.md
print("=== signal_processing requirements.txt ===")
print(get_file("signal_processing", "requirements.txt"))

# 3. ml README.md and models
print("=== ml README.md ===")
print(get_file("ml", "README.md")[:1000])

# 4. ml cnn.py and vae.py
print("=== ml src/lps_ml/model/cnn.py ===")
print(get_file("ml", "src/lps_ml/model/cnn.py")[:1000])

print("=== ml src/lps_ml/model/vae.py ===")
print(get_file("ml", "src/lps_ml/model/vae.py")[:1000])
