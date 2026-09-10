import urllib.request
import json
import os

repos = ['utils', 'signal_processing', 'ml', 'template_project']

def get_contents(repo, path=""):
    url = f"https://api.github.com/repos/labsonar/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching {repo}/{path}: {e}")
        return None

def print_tree(repo, path="", indent=0):
    items = get_contents(repo, path)
    if not items or not isinstance(items, list):
        return
    for item in items:
        prefix = "  " * indent
        print(f"{prefix}- [{item['type']}] {item['name']}")
        if item['type'] == 'dir' and indent < 2:
            print_tree(repo, item['path'], indent + 1)

if __name__ == "__main__":
    for r in repos:
        print(f"\n==================== labsonar/{r} ====================")
        print_tree(r)
