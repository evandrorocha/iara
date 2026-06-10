tex_path = r"c:\Users\ev_ro\git\IARA\presentation\apresentacao.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Frame code snippets in apresentacao.tex:")
current_frame = 0
for idx, line in enumerate(lines):
    if "\\begin{frame}" in line:
        current_frame += 1
        print(f"Frame {current_frame} (Line {idx+1}):")
        # Print next 4 lines
        for offset in range(0, 5):
            if idx + offset < len(lines):
                print(f"  {lines[idx+offset].strip()}")
        print("-" * 40)

