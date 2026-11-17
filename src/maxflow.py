"""Алгоритм Форда - Фалкерсона для задачи о максимальном потоке.

Сеть хранится списками смежности в виде массива рёбер: ребро с номером e
и обратное ему остаточное ребро e ^ 1 лежат рядом, поэтому пересчёт
остаточных пропускных способностей выполняется за O(1).
"""
from collections import deque


class FlowNetwork:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]   # adj[v] - номера рёбер, выходящих из v
        self.to = []                        # to[e]  - конец ребра e
        self.cap = []                       # cap[e] - остаточная пропускная способность
        self.augmentations = 0              # счётчик увеличивающих путей
        self.edge_checks = 0                # счётчик просмотров рёбер (базовая операция)

    def add_edge(self, u, v, c):
        self.adj[u].append(len(self.to)); self.to.append(v); self.cap.append(c)
        self.adj[v].append(len(self.to)); self.to.append(u); self.cap.append(0)

    # --- поиск увеличивающего пути -------------------------------------
    def _path_dfs(self, s, t):
        """Поиск в глубину (нерекурсивный). Возвращает parent_edge или None."""
        adj, to, cap = self.adj, self.to, self.cap
        parent = [-1] * self.n
        seen = [False] * self.n
        seen[s] = True
        pos = [0] * self.n                  # сколько рёбер вершины уже просмотрено
        stack = [s]
        checks = 0
        while stack:
            v = stack[-1]
            edges = adj[v]
            i = pos[v]
            advanced = False
            while i < len(edges):
                e = edges[i]
                i += 1
                checks += 1
                w = to[e]
                if cap[e] > 0 and not seen[w]:
                    seen[w] = True
                    parent[w] = e
                    if w == t:
                        self.edge_checks += checks
                        return parent
                    stack.append(w)
                    advanced = True
                    break
            pos[v] = i
            if not advanced:
                stack.pop()
        self.edge_checks += checks
        return None

    def _path_bfs(self, s, t):
        """Поиск в ширину - кратчайший по числу рёбер путь (Эдмондс - Карп)."""
        adj, to, cap = self.adj, self.to, self.cap
        parent = [-1] * self.n
        seen = [False] * self.n
        seen[s] = True
        queue = deque([s])
        checks = 0
        while queue:
            v = queue.popleft()
            for e in adj[v]:
                checks += 1
                w = to[e]
                if cap[e] > 0 and not seen[w]:
                    seen[w] = True
                    parent[w] = e
                    if w == t:
                        self.edge_checks += checks
                        return parent
                    queue.append(w)
        self.edge_checks += checks
        return None

    # --- основной цикл метода ------------------------------------------
    def max_flow(self, s, t, search="dfs"):
        find = self._path_dfs if search == "dfs" else self._path_bfs
        to, cap = self.to, self.cap
        flow = 0
        while True:
            parent = find(s, t)
            if parent is None:              # увеличивающего пути нет - поток максимален
                return flow
            delta = float("inf")            # остаточная пропускная способность пути
            v = t
            while v != s:
                e = parent[v]
                if cap[e] < delta:
                    delta = cap[e]
                v = to[e ^ 1]
            v = t
            while v != s:                   # увеличение потока вдоль пути
                e = parent[v]
                cap[e] -= delta
                cap[e ^ 1] += delta
                v = to[e ^ 1]
            flow += delta
            self.augmentations += 1

    def min_cut_side(self, s):
        """Множество вершин, достижимых из s в остаточной сети (сторона S разреза)."""
        seen = [False] * self.n
        seen[s] = True
        stack = [s]
        while stack:
            v = stack.pop()
            for e in self.adj[v]:
                w = self.to[e]
                if self.cap[e] > 0 and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        return seen


def build(n, edges):
    g = FlowNetwork(n)
    for u, v, c in edges:
        g.add_edge(u, v, c)
    return g
