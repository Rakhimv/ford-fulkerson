"""Построение рисунков отчёта (figures/*.png)."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from maxflow import build
from generator import bad_network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11,
    "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE, ORANGE, GRAY, INK = "#1f5fbf", "#d9730d", "#777777", "#222222"
DFS = dict(color=BLUE, marker="o", ls="-", lw=2, ms=6, label="Поиск в глубину (DFS)")
BFS = dict(color=ORANGE, marker="s", ls="--", lw=2, ms=6, label="Поиск в ширину (BFS)")

# ---------------------------------------------------------------- схемы сетей
EX_EDGES = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (2, 4, 14),
            (3, 2, 9), (3, 5, 20), (4, 3, 7), (4, 5, 4)]
EX_POS = {0: (0, 1), 1: (1.5, 2), 2: (1.5, 0), 3: (3.5, 2), 4: (3.5, 0), 5: (5, 1)}
EX_NAMES = {0: "s", 1: "1", 2: "2", 3: "3", 4: "4", 5: "t"}
R = 0.24


def draw_net(ax, pos, names, edges, labels, hl=(), dashed=(), side=None, title=None, fs=12):
    """edges: [(u, v)], labels: подписи дуг, hl: выделенные дуги (в порядке пути)."""
    for (u, v), lab in zip(edges, labels):
        a, b = np.array(pos[u], float), np.array(pos[v], float)
        d = (b - a) / np.linalg.norm(b - a)
        on = (u, v) in hl
        col = BLUE if on else ("#999999" if (u, v) in dashed else INK)
        ax.add_patch(FancyArrowPatch(a + d * R, b - d * R, arrowstyle="-|>", mutation_scale=13,
                                     lw=2.6 if on else 1.1, color=col,
                                     ls="--" if (u, v) in dashed else "-", zorder=2))
        mid = (a + b) / 2
        nrm = np.array([-d[1], d[0]]) * 0.277
        ax.text(*(mid + nrm), lab, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold" if on else "normal", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for v, (x, y) in pos.items():
        fill = "white" if side is None or not side[v] else "#cfe0f7"
        ax.add_patch(Circle((x, y), R, fc=fill, ec=INK, lw=1.3, zorder=3))
        ax.text(x, y, names[v], ha="center", va="center", fontsize=fs + 1, style="italic", zorder=5)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45); ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fs, pad=4)


def trace_example():
    """Пошаговый прогон реализации на учебной сети; возвращает список шагов."""
    g = build(6, EX_EDGES)
    steps, flow = [], 0
    while True:
        parent = g._path_dfs(0, 5)
        if parent is None:
            break
        path, v, delta = [], 5, float("inf")
        while v != 0:
            e = parent[v]; path.append(e); delta = min(delta, g.cap[e]); v = g.to[e ^ 1]
        path.reverse()
        for e in path:
            g.cap[e] -= delta; g.cap[e ^ 1] += delta
        flow += delta
        steps.append({"path": [(g.to[e ^ 1], g.to[e]) for e in path], "delta": delta, "flow": flow,
                      "f": [g.cap[2 * i + 1] for i in range(len(EX_EDGES))]})
    return steps, g.min_cut_side(0)


def fig_example():
    uv = [(u, v) for u, v, _ in EX_EDGES]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    draw_net(ax, EX_POS, EX_NAMES, uv, [str(c) for *_, c in EX_EDGES], fs=11)
    fig.savefig(os.path.join(FIG, "example_network.png")); plt.close(fig)

    steps, side = trace_example()
    cols = 2
    rows = (len(steps) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 2.55 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, st in enumerate(steps):
        fwd = set(st["path"]); back = {(v, u) for u, v in st["path"]}
        hl = [e for e in uv if e in fwd or e in back]
        names = " → ".join([EX_NAMES[st["path"][0][0]]] + [EX_NAMES[v] for _, v in st["path"]])
        draw_net(axes.flat[k], EX_POS, EX_NAMES, uv,
                 [f"{f}/{c}" for f, (*_, c) in zip(st["f"], EX_EDGES)], hl=hl,
                 title=f"Шаг {k + 1}: {names},  δ = {st['delta']},  |f| = {st['flow']}", fs=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "example_steps.png")); plt.close(fig)

    last = steps[-1]
    cut = [(u, v) for u, v in uv if side[u] and not side[v]]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    draw_net(ax, EX_POS, EX_NAMES, uv, [f"{f}/{c}" for f, (*_, c) in zip(last["f"], EX_EDGES)],
             hl=cut, side=side, fs=11)
    fig.savefig(os.path.join(FIG, "example_mincut.png")); plt.close(fig)

    info = {"steps": [{"path": [EX_NAMES[s["path"][0][0]]] + [EX_NAMES[v] for _, v in s["path"]],
                       "delta": s["delta"], "flow": s["flow"]} for s in steps],
            "S": [EX_NAMES[v] for v in range(6) if side[v]],
            "T": [EX_NAMES[v] for v in range(6) if not side[v]],
            "cut": [[EX_NAMES[u], EX_NAMES[v], c] for u, v, c in EX_EDGES if side[u] and not side[v]]}
    with open(os.path.join(RES, "example_trace.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))


def fig_bad():
    pos = {0: (0, 1), 1: (1.6, 2), 2: (1.6, 0), 3: (3.2, 1)}
    names = {0: "s", 1: "a", 2: "b", 3: "t"}
    uv = [(u, v) for u, v, _ in bad_network(1)]
    lab = ["U", "U", "1", "U", "U"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))
    draw_net(axes[0], pos, names, uv, lab, title="а) исходная сеть", fs=10)
    draw_net(axes[1], pos, names, uv, lab, hl=[(0, 1), (1, 2), (2, 3)],
             title="б) нечётные шаги: s → a → b → t", fs=10)
    draw_net(axes[2], pos, names, uv, lab, hl=[(0, 2), (1, 2), (1, 3)],
             title="в) чётные шаги: s → b → a → t", fs=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bad_network.png")); plt.close(fig)


# ---------------------------------------------------------------- графики
def legend_below(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=10, handlelength=2.6)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def col(rows, k):
    return np.array([r[k] for r in rows])


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def fig_scaling(name, out, xlabel):
    rows = load(name)
    n = col(rows, "n")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, key, yl in ((axes[0], "time", "Время работы, с"),
                        (axes[1], "aug", "Число увеличивающих путей")):
        ax.plot(n, col(rows, key + "_dfs"), **DFS)
        ax.plot(n, col(rows, key + "_bfs"), **BFS)
        if key == "aug":
            ax.plot(n, col(rows, "flow"), color=GRAY, ls=":", lw=2, label="Величина потока |f*|")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(n); ax.set_xticklabels([str(int(v)) for v in n])
        ax.set_xlabel(xlabel); ax.set_ylabel(yl)
    legend_below(fig, axes[1])
    fig.savefig(os.path.join(FIG, out)); plt.close(fig)


def fig_capacity():
    rows = load("capacity.csv")
    U = col(rows, "U")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    axes[0].plot(U, col(rows, "aug_dfs"), **DFS)
    axes[0].plot(U, col(rows, "aug_bfs"), **BFS)
    axes[0].plot(U, col(rows, "flow"), color=GRAY, ls=":", lw=2, label="Величина потока |f*|")
    axes[0].set_ylabel("Число увеличивающих путей")
    axes[1].plot(U, col(rows, "time_dfs"), **DFS)
    axes[1].plot(U, col(rows, "time_bfs"), **BFS)
    axes[1].set_ylabel("Время работы, с")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Максимальная пропускная способность U")
    legend_below(fig, axes[0])
    fig.savefig(os.path.join(FIG, "capacity.png")); plt.close(fig)


def fig_bad_time():
    rows = load("bad.csv")
    U = col(rows, "U")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    adv = dict(DFS, label="Наихудший выбор путей")
    for ax, key, yl in ((axes[0], "aug", "Число увеличивающих путей"),
                        (axes[1], "time", "Время работы, с")):
        ax.plot(U, col(rows, key + "_adv"), **adv)
        ax.plot(U, col(rows, key + "_bfs"), **BFS)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Пропускная способность U"); ax.set_ylabel(yl)
    legend_below(fig, axes[0])
    fig.savefig(os.path.join(FIG, "bad_time.png")); plt.close(fig)


def fig_ratio():
    """Отношение числа просмотров рёбер к априорной оценке E·|f*|."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for name, lab, st in (("sparse.csv", "Разреженные сети, m = 5n", dict(color=BLUE, marker="o", ls="-")),
                          ("dense.csv", "Плотные сети, m = n²/4", dict(color=ORANGE, marker="s", ls="--"))):
        rows = load(name)
        ax.plot(col(rows, "n"), col(rows, "checks_dfs") / (2 * col(rows, "m") * col(rows, "flow")),
                lw=2, ms=6, label=lab, **st)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_ylim(0, None)
    ax.set_xlabel("Число вершин n"); ax.set_ylabel("Просмотров дуг / (2m · |f*|)")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ratio.png")); plt.close(fig)


def summary():
    out = {}
    for name in ("sparse.csv", "dense.csv"):
        rows = load(name); n = col(rows, "n")
        out[name] = {k: round(slope(n, col(rows, k)), 2)
                     for k in ("time_dfs", "time_bfs", "aug_dfs", "aug_bfs", "flow", "checks_dfs", "checks_bfs")}
    rows = load("capacity.csv"); U = col(rows, "U")
    out["capacity.csv"] = {k: round(slope(U[1:], col(rows, k)[1:]), 2)
                           for k in ("time_dfs", "time_bfs", "aug_dfs", "aug_bfs", "flow")}
    rows = load("bad.csv"); U = col(rows, "U")
    out["bad.csv"] = {"time_adv": round(slope(U, col(rows, "time_adv")), 2)}
    with open(os.path.join(RES, "slopes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    fig_example()
    fig_bad()
    if os.path.exists(os.path.join(RES, "dense.csv")):
        fig_scaling("sparse.csv", "sparse.png", "Число вершин n (m = 5n)")
        fig_scaling("dense.csv", "dense.png", "Число вершин n (m = n²/4)")
        fig_capacity()
        fig_bad_time()
        fig_ratio()
        summary()
