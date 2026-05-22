"""
并发请求测试：验证多请求是否真正并行处理。

测试 A：缓存命中（极快请求）  → 验证并发连接数
测试 B：缓存 miss（需下载瓦片）→ 验证 I/O 并行加速比
"""

import threading
import time
import requests
import sys

BASE_URL = "http://localhost:19091"

# 不同坐标 → 不同 hash → 不同缓存文件 → 可产生缓存 miss
BOXES = [
    {"x": 49.25, "y": 45.36, "width": 1.91, "height": 2.59},
    {"x": 30.00, "y": 30.00, "width": 2.00, "height": 2.00},
    {"x": 60.00, "y": 55.00, "width": 1.80, "height": 2.20},
    {"x": 20.00, "y": 20.00, "width": 1.50, "height": 1.80},
]


def make_payload(box: dict) -> dict:
    return {
        "tasks": [{"data": {"image": "https://mdi.hkust-gz.edu.cn/wsi/metaservice/api/sliceInfo/openslide/TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs"}}],
        "params": {
            "login": None, "password": None,
            "context": {
                "result": [{
                    "original_width": 95488,
                    "original_height": 81920,
                    "image_rotation": 0,
                    "value": {**box, "rectanglelabels": ["cell"]},
                    "type": "rectanglelabels",
                    "origin": "manual"
                }],
                "cur_scale": 0.030145485221434256
            }
        },
        "task_id": "106843",
        "img_type": "svs"
    }


def send_request(req_id: int, payload: dict, results: dict):
    t0 = time.time()
    try:
        resp = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=120)
        t1 = time.time()
        results[req_id] = {"start": t0, "end": t1, "elapsed": round(t1 - t0, 3),
                           "ok": resp.status_code == 200, "status": resp.status_code}
    except Exception as e:
        t1 = time.time()
        results[req_id] = {"start": t0, "end": t1, "elapsed": round(t1 - t0, 3), "ok": False, "error": str(e)}


def print_results(results: dict, t_start: float):
    for i in sorted(results, key=lambda x: results[x]["start"]):
        r = results[i]
        rel_start = round(r["start"] - t_start, 3)
        bar = "█" * int(r["elapsed"] * 20)
        status = "✓" if r["ok"] else f"✗({r.get('status', r.get('error', '?'))})"
        print(f"  [{i+1}] {status}  +{rel_start:5.3f}s │{bar}│ {r['elapsed']}s")


def check_overlap(results: dict, n: int) -> int:
    overlapping = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if results[i]["start"] < results[j]["end"] and results[j]["start"] < results[i]["end"]
    )
    return overlapping


def run_test(label: str, payloads: list, concurrent: bool) -> float:
    n = len(payloads)
    print(f"\n{'='*60}")
    print(f"{'并发' if concurrent else '顺序'}执行 {n} 个请求  [{label}]")
    print(f"{'='*60}")

    results = {}
    t_start = time.time()

    if concurrent:
        threads = [threading.Thread(target=send_request, args=(i, payloads[i], results)) for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()
    else:
        for i in range(n):
            send_request(i, payloads[i], results)

    total = round(time.time() - t_start, 3)
    print_results(results, t_start)

    if concurrent:
        overlaps = check_overlap(results, n)
        total_pairs = n * (n - 1) // 2
        print(f"\n  时间重叠对数: {overlaps}/{total_pairs}  {'← 并行确认 ✓' if overlaps > 0 else '← 无重叠'}")

    all_ok = all(r["ok"] for r in results.values())
    print(f"  总耗时: {total}s  {'全部成功' if all_ok else '有失败请求!'}")
    return total


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n = min(n, len(BOXES))

    print(f"目标服务: {BASE_URL}   并发数 N={n}")

    # ── 测试 A：缓存命中（瓦片已缓存，只测 GPU 推理并发）──────────────────
    print("\n\n【测试 A】缓存命中场景（排除 I/O，测推理并发）")
    warm_payload = make_payload(BOXES[0])
    print("  预热中...")
    requests.post(f"{BASE_URL}/api/predict", json=warm_payload, timeout=120)
    cached_payloads = [make_payload(BOXES[0])] * n   # 同一坐标 → 全命中缓存

    seq_a  = run_test("缓存命中-顺序", cached_payloads, concurrent=False)
    conc_a = run_test("缓存命中-并发", cached_payloads, concurrent=True)

    # ── 测试 B：缓存 miss（每个请求下载不同区域，测 I/O 并行）──────────────
    print("\n\n【测试 B】缓存 miss 场景（不同坐标 → 各自下载瓦片 → 测 I/O 并行）")
    miss_payloads = [make_payload(BOXES[i]) for i in range(n)]

    seq_b  = run_test("缓存miss-顺序", miss_payloads, concurrent=False)
    conc_b = run_test("缓存miss-并发", miss_payloads, concurrent=True)

    # ── 汇总 ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    def speedup_label(seq, conc):
        sp = round(seq / conc, 2) if conc > 0 else 0
        if sp >= 1.5:   tag = "✓ 明显并行加速"
        elif sp >= 1.1: tag = "~ 轻微并行效果"
        else:           tag = "≈ 基本串行（正常，受 GPU Lock）"
        return f"{sp}x  {tag}"

    print(f"  测试A 顺序={seq_a}s  并发={conc_a}s  加速比={speedup_label(seq_a, conc_a)}")
    print(f"  测试B 顺序={seq_b}s  并发={conc_b}s  加速比={speedup_label(seq_b, conc_b)}")
    print()
    print("  说明：")
    print("  - 测试A（缓存命中）：请求极快，线程调度开销与请求耗时相当，加速比有限")
    print("  - 测试B（缓存miss）：I/O 下载阶段真正并行，加速比应明显 >1")
    print("  - 时间窗口有重叠 = Gunicorn gthread 并发连接工作正常")


if __name__ == "__main__":
    main()
