import psutil
import time

logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

print(f"Monitorando uso de CPU durante 3 segundos em {logical_cores} threads ({physical_cores} núcleos físicos)...")

# Manter o mesmo dicionário de instâncias de Process
proc_map = {}
for p in psutil.process_iter(['pid', 'name']):
    try:
        p.cpu_percent(interval=None) # baseline
        proc_map[p.pid] = p
    except:
        pass

time.sleep(3.0)

procs = []
for pid, p in list(proc_map.items()):
    try:
        if not p.is_running():
            continue
        raw_cpu = p.cpu_percent(interval=None)
        norm_cpu = raw_cpu / logical_cores
        mem_mb = (p.memory_info().rss / (1024 * 1024))
        name = p.name()
        if name != "System Idle Process" and norm_cpu > 0.05:
            procs.append({
                'pid': pid,
                'name': name,
                'cpu_sys_pct': norm_cpu,
                'cpu_core_pct': raw_cpu,
                'mem_mb': mem_mb,
            })
    except:
        pass

procs.sort(key=lambda x: x['cpu_sys_pct'], reverse=True)

total_sys_cpu = psutil.cpu_percent(interval=None)

print(f"\nUso Total Global da CPU no momento: {total_sys_cpu:.1f}%\n")
print(f"{'Nome do Processo':<30} {'PID':<8} {'% CPU Total':<15} {'% 1 Núcleo':<12} {'RAM (MB)':<10}")
print("-" * 80)
for p in procs[:25]:
    print(f"{p['name']:<30} {p['pid']:<8} {p['cpu_sys_pct']:<15.2f} {p['cpu_core_pct']:<12.1f} {p['mem_mb']:<10.1f}")

print("\n--- Agrupado por Aplicativo ---")
grouped = {}
for p in procs:
    n = p['name']
    grouped.setdefault(n, {'cpu': 0.0, 'mem': 0.0, 'count': 0})
    grouped[n]['cpu'] += p['cpu_sys_pct']
    grouped[n]['mem'] += p['mem_mb']
    grouped[n]['count'] += 1

grouped_list = sorted(grouped.items(), key=lambda x: x[1]['cpu'], reverse=True)
print(f"{'Aplicativo / Processo':<30} {'Instâncias':<12} {'% CPU Total':<15} {'RAM Total (MB)'}")
print("-" * 75)
for name, data in grouped_list[:15]:
    print(f"{name:<30} {data['count']:<12} {data['cpu']:<15.2f} {data['mem']:<10.1f}")
