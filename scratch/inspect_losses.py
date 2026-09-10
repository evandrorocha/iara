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
            return str(data)
    except Exception as e:
        return f"Error: {e}"

print("=== sonar_loss.py ===")
print(get_file("ml", "src/lps_ml/utils/sonar_loss.py")[:2000])

print("\n=== audio_vae.py ===")
print(get_file("ml", "src/lps_ml/model/audio_vae.py")[:2000])
