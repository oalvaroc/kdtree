
import random
import time
from kdtree import KDTree, euclid2

def gera_pontos(num_pontos, dimensoes, seed=42):
    random.seed(seed)
    return [[random.uniform(0, 100) for _ in range(dimensoes)] for _ in range(num_pontos)]


def brute_force_metodo(pontos, target):
    dist_minima = float("inf")
    ponto_mais_perto = None
    for ponto in pontos:
        dist = euclid2(ponto, target)
        if dist < dist_minima:
            dist_minima = dist
            ponto_mais_perto = ponto
    return ponto_mais_perto, dist_minima ** 0.5


def mede_tempo(num_pontos, dimensoes):
    pontos = gera_pontos(num_pontos, dimensoes)
    target = [random.uniform(0, 100) for _ in range(dimensoes)]

    kd_tree = KDTree(dimensoes)
    for ponto in pontos:
        kd_tree.insert(ponto)

    start = time.time()
    brute_force_metodo(pontos, target)
    brute_time = time.time() - start

    start = time.time()
    kd_tree.nearest_neighbour(target)
    kd_time = time.time() - start

    return brute_time, kd_time


def experimento():
    tamanhos = [10, 100, 1_000, 10_000, 100_000]
    dimensoes = 3
    results = []

    print("N Points | Brute Force Time (s) | KD-Tree Time (s)")
    print("-" * 40)

    for tamanho in tamanhos:
        brute_time, kd_time = mede_tempo(tamanho, dimensoes)
        results.append((tamanho, brute_time, kd_time))
        print(f"{tamanho:7d} | {brute_time:18.6f} | {kd_time:14.6f}")

    return results



experimento()
