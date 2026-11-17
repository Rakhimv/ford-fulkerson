"""Вычислительные эксперименты. Результаты пишутся в results/*.csv."""
import csv
import os
import sys
import time

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from maxflow import build
from generator import random_network, bad_network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

SEEDS = 10      # число случайных сетей на одну точку
REPEATS = 3     # повторов замера на одну сеть (берётся минимум)


def measure(n, edges, search):
    best = None
    for _ in range(REPEATS):
        g = build(n, edges)
        t0 = time.perf_counter()
        f = g.max_flow(0, n - 1, search)
        dt = time.perf_counter() - t0
        if best is None or dt < best[0]:
            best = (dt, f, g.augmentations, g.edge_checks)
        if dt > 0.3:                        # долгие запуски не повторяем
            break
    return best


def run_point(n, m, U, base_seed, SEEDS=SEEDS):
    """Среднее по SEEDS сетям для обоих вариантов поиска пути."""
    acc = {"dfs": [0.0, 0, 0], "bfs": [0.0, 0, 0]}
    flow_sum = 0
    for k in range(SEEDS):
        edges = random_network(n, m, U, base_seed + k)
        flows = {}
        for search in ("dfs", "bfs"):
            dt, f, aug, chk = measure(n, edges, search)
            flows[search] = f
            a = acc[search]
            a[0] += dt; a[1] += aug; a[2] += chk
        assert flows["dfs"] == flows["bfs"], "варианты дали разный поток"
        flow_sum += flows["dfs"]
    row = {"n": n, "m": m, "U": U, "flow": flow_sum / SEEDS}
    for search in ("dfs", "bfs"):
        a = acc[search]
        row[f"time_{search}"] = a[0] / SEEDS
        row[f"aug_{search}"] = a[1] / SEEDS
        row[f"checks_{search}"] = a[2] / SEEDS
    return row


def save(name, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(name, "готово")
    for r in rows:
        print("  ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def correctness():
    """Сверка с эталонной реализацией networkx и проверка теоремы о разрезе."""
    import random
    rng = random.Random(2026)
    total = 300
    for i in range(total):
        n = rng.randint(2, 40)
        m_max = (n - 1) * (n - 2) + 1
        m = rng.randint(n - 1, min(m_max, 6 * n))
        U = rng.choice([1, 5, 100, 10**6])
        edges = random_network(n, m, U, 10_000 + i)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, v, c in edges:
            G.add_edge(u, v, capacity=c)
        ref = nx.maximum_flow_value(G, 0, n - 1)
        for search in ("dfs", "bfs"):
            g = build(n, edges)
            f = g.max_flow(0, n - 1, search)
            assert f == ref, (i, search, f, ref)
            side = g.min_cut_side(0)
            cut = sum(c for u, v, c in edges if side[u] and not side[v])
            assert cut == f, (i, search, cut, f)
    save("correctness.csv", [{"tests": total, "passed": total}])


def exp_sparse():
    rows = [run_point(n, 5 * n, 100, 100) for n in (100, 200, 400, 800, 1600, 3200, 6400)]
    save("sparse.csv", rows)


def exp_dense():
    rows = [run_point(n, n * n // 4, 100, 200) for n in (25, 50, 100, 200, 400)]
    save("dense.csv", rows)


def exp_capacity():
    rows = [run_point(100, 1000, U, 300, SEEDS=5) for U in (1, 10, 100, 10**3, 10**4, 10**5)]
    save("capacity.csv", rows)


def adversarial_bad(U):
    """«Плохая» сеть с наихудшим выбором путей: s-a-b-t и s-b-a-t поочерёдно."""
    g = build(4, bad_network(U))
    cap = g.cap
    paths = ((0, 4, 8), (2, 5, 6))          # номера рёбер двух чередующихся путей
    t0 = time.perf_counter()
    flow = 0
    k = 0
    while True:
        p = paths[k & 1]
        delta = min(cap[p[0]], cap[p[1]], cap[p[2]])
        if delta == 0:
            break
        for e in p:
            cap[e] -= delta
            cap[e ^ 1] += delta
        flow += delta
        g.augmentations += 1
        k += 1
    flow += g.max_flow(0, 3, "dfs")         # убеждаемся, что путей больше нет
    return time.perf_counter() - t0, flow, g.augmentations


def exp_bad():
    rows = []
    for U in (10**2, 10**3, 10**4, 10**5, 10**6):
        t_adv, f_adv, a_adv = min(adversarial_bad(U) for _ in range(REPEATS))
        t_bfs, f_bfs, a_bfs, _ = measure(4, bad_network(U), "bfs")
        t_dfs, f_dfs, a_dfs, _ = measure(4, bad_network(U), "dfs")
        assert f_adv == f_bfs == f_dfs == 2 * U
        rows.append({"U": U, "flow": f_adv,
                     "aug_adv": a_adv, "time_adv": t_adv,
                     "aug_dfs": a_dfs, "time_dfs": t_dfs,
                     "aug_bfs": a_bfs, "time_bfs": t_bfs})
    save("bad.csv", rows)


if __name__ == "__main__":
    correctness()
    exp_bad()
    exp_capacity()
    exp_sparse()
    exp_dense()
