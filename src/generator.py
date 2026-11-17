"""Генератор входных данных: случайные транспортные сети и «плохая» сеть."""
import random


def random_network(n, m, U, seed):
    """Случайная сеть: n вершин, m дуг, целые пропускные способности из [1, U].

    Исток - вершина 0, сток - вершина n-1. Сначала строится случайный
    гамильтонов путь из истока в сток (гарантирует достижимость стока и
    участие каждой вершины в сети), затем добавляются случайные дуги без
    петель и кратных дуг, пока их общее число не станет равным m.
    В исток дуги не входят, из стока - не выходят.
    """
    assert n - 1 <= m <= (n - 1) * (n - 2) + 1
    rng = random.Random(seed)
    inner = list(range(1, n - 1))
    rng.shuffle(inner)
    order = [0] + inner + [n - 1]
    used = set()
    edges = []
    for u, v in zip(order, order[1:]):
        used.add((u, v))
        edges.append((u, v, rng.randint(1, U)))
    while len(edges) < m:
        u = rng.randrange(0, n - 1)          # не сток
        v = rng.randrange(1, n)              # не исток
        if u == v or (u, v) in used:
            continue
        used.add((u, v))
        edges.append((u, v, rng.randint(1, U)))
    rng.shuffle(edges)
    return edges


def bad_network(U):
    """Классическая «плохая» сеть из 4 вершин: s=0, a=1, b=2, t=3."""
    return [(0, 1, U), (0, 2, U), (1, 2, 1), (1, 3, U), (2, 3, U)]
