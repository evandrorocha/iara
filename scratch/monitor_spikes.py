import psutil
import time

logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

print(f"Monitorando picos de CPU durante 5 amostragens (1s cada)...")
print(f"Sistema: {physical_cores} núcleos físicos, {logical_cores} threads lógicas\n")

for sample in range(1, 6):
    sys_cpu = psutil.cpu_percent(interval=1.0)
    top_procs = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            # percent over interval
            raw = p.cpu_percent(interval=None)
            pct = raw / logical_cores
            if pct > 0.4:
                top_procs.append((pct, p.info['name'], p.info['pid']))
        except:
            pass
    top_procs.sort(reverse=True)
    print(f"[Amostra {sample}/5] CPU Global: {sys_cpu:.1f}%")
    for pct, name, pid in top_procs[:6]:
        print(f"    - {name:<25} (PID {pid:<7}): {pct:.1f}% global ({pct * logical_cores:.1f}% de 1 núcleo)")
    print()
