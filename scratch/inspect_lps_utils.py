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

files = ['quantities.py', 'hashable.py', 'prefered_number.py', 'utils.py', 'unity.py', 'subprocess.py', 'log.py']
for f in files:
    content = get_file('utils', f'src/lps_utils/{f}')
    print(f"=== {f} ({len(content)} chars) ===")
    lines = content.splitlines()
    print("\n".join(lines[:35]))
    print("...")
