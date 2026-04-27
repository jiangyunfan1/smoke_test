import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 如果是早期重构后代码，改成 process_name = connector，由于进程名发生了从 mindie_llm_backend -> connector -> mindie_llm_backend 变更，需要根据实际情况填入合适进程名称
process_name = "VLLMEngineCor" 

def parse_timestamp(line):
    try:
        return datetime.strptime(line.split('\n')[0].strip(), '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None

def parse_free_m_section(lines):
    mem_info = {}
    for line in lines:
        if line.startswith("Mem:"):
            parts = line.split()
            mem_info = {
                'total': int(parts[1]),
                'used': int(parts[2]),
                'free': int(parts[3]),
                'shared': int(parts[4]),
                'buff/cache': int(parts[5]),
                'available': int(parts[6]),
            }
            break
    return mem_info


def parse_smem_section(lines):
    """
    解析 smem 输出为 dict：{ (pid, command): {swap, uss, pss, rss} }
    """
    result = {}
    for line in lines:
        if line.strip().startswith("PID") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 7:
            try:
                pid = int(parts[0])
                command = parts[2]
                swap = int(parts[-4])
                uss  = int(parts[-3])
                pss  = int(parts[-2])
                rss  = int(parts[-1])
                result[(pid, command)] = {
                    'Swap': swap,
                    'USS': uss,
                    'PSS': pss,
                    'RSS': rss
                }
            except ValueError:
                continue
    return result



def extract_entries_from_log(logfile):
    with open(logfile, 'r', encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("======================")
    records = []
    for block in blocks[:-1]:
        lines = block.strip().split('----------------------')
        if len(lines) != 3:
            continue
        timestamp = parse_timestamp(lines[0])
        if not timestamp:
            continue
        try:
            # reverse find separators
            free_m_lines = lines[1].strip()
            smem_lines = lines[2].strip()
            free_data = parse_free_m_section(free_m_lines.splitlines())
            smem_data = parse_smem_section(smem_lines.splitlines())
            records.append((timestamp, free_data, smem_data))
        except Exception as e:
            print(f"[WARN] Failed to parse block at {timestamp}: {e}")
            continue

    return records

def get_mindie_pids(logfile):
    with open(logfile, 'r', encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("======================")
    pids = []

    for block in blocks[1:2]:
        lines = block.strip().split('----------------------')
        if len(lines) != 3:
            continue

        for line in lines[0].split('\n'):
            if process_name in line:
                pids.append(int(line.split()[1]))
        return pids


def diff_dicts(d1, d2):
    return {k: d2[k] - d1.get(k, 0) for k in d2 if isinstance(d2[k], int) and k in d1}


def format_kb(kb):
    return f"{kb / 1024/1024:.2f} GB" if abs(kb) >= 1024 else f"{kb} KB"


def summarize_smem_diff(smem1, smem2):
    """
    比较两次 smem 结果，输出新增、删除、变化的进程及其内存变化摘要
    """
    all_keys = set(smem1) | set(smem2)
    added = []
    removed = []
    changed = []

    rss_increase_total = 0
    rss_decrease_total = 0
    pss_increase_total = 0
    pss_decrease_total = 0

    pss_total_delta = 0

    for key in all_keys:
        val1 = smem1.get(key)
        val2 = smem2.get(key)

        if val1 and not val2:
            removed.append((key, val1))
        elif val2 and not val1:
            added.append((key, val2))
        else:
            delta = {
                k: val2[k] - val1[k] for k in ('Swap', 'USS', 'PSS', 'RSS')
            }
            if any(delta.values()):
                changed.append((key, delta))
                rss_delta = delta['RSS']
                pss_delta = delta['PSS']
                pss_total_delta += delta['PSS']
                # if rss_delta > 0:
                rss_increase_total += rss_delta
                # elif rss_delta < 0:
                rss_decrease_total += -rss_delta
                # if pss_delta > 0:
                pss_increase_total += pss_delta
                # elif pss_delta < 0:
                pss_decrease_total += -pss_delta

    # 排序变化列表，按 RSS 增量倒序
    changed_sorted = sorted(
        changed, key=lambda x: x[1]['PSS'], reverse=True
    )

    print("\n=== smem 内存变化进程（按 PSS 增量排序 Top 10） ===")
    for (pid, cmd), delta in changed_sorted[:30]:
        print(f"{pid:<7} {cmd:<28} ΔPSS: {format_kb(delta['PSS']):>8}")


def fit_memory_rate(timestamps, mem, last_hour=1):
    """获取内存增长率"""
    n_point = int(last_hour * 60) # 每小时60个点
    timestamps = timestamps[-n_point:]
    mem = mem[-n_point:]
    start_time = timestamps[0]
    x = np.array([(t - start_time).total_seconds() for t in timestamps])
    y = np.array(mem)

    # 最小二乘法拟合直线 y = kx + b
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # 输出结果
    growth_per_hour = k * 3600 * 1024  # MB/h
    return growth_per_hour


def get_memory_usages(records, pid):
    if isinstance(pid, int):
        pid = [pid]
    timestamps = [r[0] for r in records]
    # 获取 free -m 中的 used 字段
    free_mem = [r[1].get('used', 0)/1024 for r in records]

    smem_pss = []
    smem_rss = []
    for timestamp, free_data, smem_data in records:
        pss = 0
        rss = 0
        for pid_name, mem in smem_data.items():

            if pid_name[0] in pid:
                # smem_pss.append(mem.get('PSS', 0) / 1024 / 1024)
                # smem_rss.append(mem.get('RSS', 0) / 1024 / 1024)
                pss += mem.get('PSS', 0) / 1024 / 1024
                rss += mem.get('RSS', 0) / 1024 / 1024
        # timestamps.append(timestamp)
        smem_pss.append(pss)
        smem_rss.append(rss)
    return timestamps, free_mem, smem_pss


def plot_memory_usage(timestamps, free_mem, smem_pss_daemon, smem_pss_connector, last_hours=1.0):
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    rate = fit_memory_rate(timestamps, free_mem, last_hours)
    label = f"free -m: used | 整机\n最后 {last_hours:.1f}h 内存增长量: {rate:.2f} MB/h"
    axs[0].plot(timestamps, free_mem, ".-", label=label, color="blue")
    axs[0].set_ylabel("Whole Host Used Mem (GB)")
    axs[0].legend(loc='upper left')
    # axs[0].set_ylim([0, 200])
    axs[0].grid(True)

    rate = fit_memory_rate(timestamps, smem_pss_daemon, last_hours)
    title = f"单个 daemon 进程内存\n最后 {last_hours:.1f}h 内存增长量: {rate:.2f} MB/h"
    axs[1].plot(timestamps, smem_pss_daemon, ".-", label=title, color="green")
    axs[1].set_ylabel("One process PSS Total (GB)")
    axs[1].legend(loc='upper left')
    # axs[1].set_ylim([0, 2*max(smem_pss)])
    axs[1].grid(True)

    rate = fit_memory_rate(timestamps, smem_pss_connector, last_hours)
    title = f"所有 backend_connector 进程内存之和\n最后 {last_hours:.1f}h 内存增长量: {rate:.2f} MB/h"
    axs[2].plot(timestamps, smem_pss_connector, ".-", label=title, color="brown")
    axs[2].set_ylabel("All process PSS Total (GB)")
    axs[2].set_xlabel("Timestamp")
    axs[2].legend(loc='upper left')
    # axs[2].set_ylim([0, 2*max(smem_pss)])
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def main(logfile, pid):
    pid = int(pid.strip())
    records = extract_entries_from_log(logfile)
    mindie_pids = get_mindie_pids(logfile)
    # print(mindie_pids)
    if len(records) < 2:
        print("日志条目不足两条，无法计算内存变化。")
        return

    t0, free0, smem0 = records[0]
    t1, free1, smem1 = records[-1]

    print(f"\n时间范围：{t0} -> {t1}")
    print("\n📊 [free -m] 总体内存变化（单位：MB）：")
    free_diff = diff_dicts(free0, free1)
    for k, v in free_diff.items():
        print(f"  {k:>12}: {v:<6} MB")

    print("\n📊 [smem] 各命令 USS/PSS/RSS 增量：")
    summarize_smem_diff(smem0, smem1)

    timestamps, free_mem, smem_pss_daemon = get_memory_usages(records, pid)
    timestamps, free_mem, smem_pss_connector = get_memory_usages(records, mindie_pids)

    plot_memory_usage(timestamps, free_mem, smem_pss_daemon, smem_pss_connector, last_hours=2)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python analyze_mem_log.py <日志文件名> <PID>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

