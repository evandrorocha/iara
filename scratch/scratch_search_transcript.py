import json

log_file = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\b9e437c3-f2d4-4681-96ac-ed18adc3bd7b\.system_generated\logs\transcript.jsonl"
print(f"Reading log file: {log_file}")

with open(log_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            content = str(data.get("content", ""))
            tool_calls = str(data.get("tool_calls", ""))
            
            if "pdflatex" in tool_calls.lower() or "pdflatex" in content.lower():
                print(f"Line {i} matches pdflatex:")
                print(f"  Content: {content[:200]}")
                print(f"  Tool calls: {tool_calls[:400]}")
                print("=" * 80)
        except Exception as e:
            pass




