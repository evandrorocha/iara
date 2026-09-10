import urllib.request
import json
import base64

def get_file(repo, path):
    url = f"https://api.github.com/repos/labsonar/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return [x['name'] for x in data]
            if "content" in data:
                return base64.b64decode(data["content"]).decode('utf-8', errors='replace')
            return str(data)
    except Exception as e:
        return f"Error: {e}"

print("=== lps_ml/core contents ===")
print(get_file("ml", "src/lps_ml/core"))

print("=== lps_ml/utils contents ===")
print(get_file("ml", "src/lps_ml/utils"))

print("=== lps_ml/visualization contents ===")
print(get_file("ml", "src/lps_ml/visualization"))

# Fetch test_iara.py, test_4classes.py, svdd.py, and one iemanja file
for f in [("test/test_iara.py", "scratch/ml_test_iara.py"),
         ("test/test_4classes.py", "scratch/ml_test_4classes.py"),
         ("src/lps_ml/model/svdd.py", "scratch/ml_svdd.py"),
         ("app/iemanja_classifier.py", "scratch/ml_iemanja.py")]:
    content = get_file("ml", f[0])
    if isinstance(content, str) and not content.startswith("Error"):
        with open(f[1], "w", encoding="utf-8") as out:
            out.write(content)
        print(f"Saved {f[1]} ({len(content)} bytes)")
    else:
        print(f"Failed {f[0]}: {content[:100]}")
