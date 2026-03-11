from __future__ import annotations

import argparse
import math
import os
import random
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


def parse_tsp(path: str) -> List[Tuple[float, float]]:
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        in_node_section = False
        for line in f:
            line = line.strip()
            if line.upper().startswith("NODE_COORD_SECTION"):
                in_node_section = True
                continue
            if not in_node_section:
                continue
            if line == "EOF" or line == "":
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                except ValueError:
                    continue
                coords.append((x, y))
    return coords


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def total_distance(tour: List[int], coords: List[Tuple[float, float]]) -> float:
    n = len(tour)
    if n == 0:
        return 0.0
    dist = 0.0
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % n]]
        dist += euclidean(a, b)
    return dist


def nearest_neighbor(coords: List[Tuple[float, float]], start: int = 0) -> List[int]:
    n = len(coords)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda j: euclidean(coords[last], coords[j]))
        tour.append(next_node)
        unvisited.remove(next_node)
    return tour


def two_opt(tour: List[int], coords: List[Tuple[float, float]]) -> List[int]:
    improved = True
    n = len(tour)
    best = tour[:]
    best_dist = total_distance(best, coords)
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                new_tour = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                new_dist = total_distance(new_tour, coords)
                if new_dist + 1e-12 < best_dist:
                    best = new_tour
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best


def canonical_tour(tour: List[int]) -> Tuple[int, ...]:
    n = len(tour)
    if n == 0:
        return tuple()
    min_idx = tour.index(min(tour))
    forward = tuple(tour[min_idx:] + tour[:min_idx])
    reverse_tour = tour[::-1]
    min_idx_r = reverse_tour.index(min(reverse_tour))
    reverse = tuple(reverse_tour[min_idx_r:] + reverse_tour[:min_idx_r])
    return min(forward, reverse)


def tour_edges(tour: List[int]) -> Set[frozenset]:
    return {frozenset((tour[i], tour[(i + 1) % len(tour)])) for i in range(len(tour))}


def diversity_edges(a: List[int], b: List[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    e_a = tour_edges(a)
    e_b = tour_edges(b)
    common = len(e_a.intersection(e_b))
    return 1.0 - (common / n)


@dataclass
class CBGAConfig:
    population_size: int = 40
    generations: int = 160
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    tournament_k: int = 3
    min_diversity: float = 0.2
    p_ls_child: float = 0.1
    apply_2opt_on_new: bool = True
    seed: int = 19


class CBGATSPSolver:
    def __init__(self, config: CBGAConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def _random_chromosome(self, n: int) -> List[int]:
        chrom = list(range(n))
        self.rng.shuffle(chrom)
        return chrom

    def _fitness(self, coords: List[Tuple[float, float]], chrom: List[int]) -> float:
        return total_distance(chrom, coords)

    def _tournament(self, pop: List[List[int]], fitness: List[float]) -> List[int]:
        idxs = self.rng.sample(range(len(pop)), self.config.tournament_k)
        best_idx = min(idxs, key=lambda i: fitness[i])
        return pop[best_idx][:]

    def _ox(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n
        child[a : b + 1] = p1[a : b + 1]
        fill = [g for g in p2 if g not in child]
        ptr = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    def _mutate(self, chrom: List[int]) -> None:
        if self.rng.random() < self.config.mutation_rate:
            i, j = sorted(self.rng.sample(range(len(chrom)), 2))
            chrom[i], chrom[j] = chrom[j], chrom[i]

    def solve(self, coords: List[Tuple[float, float]]) -> Dict[str, Any]:
        n = len(coords)
        if n == 0:
            return {
                "best_tour": [],
                "best_distance": 0.0,
                "history": [],
                "improvements": [],
                "time_wall": 0.0,
                "time_cpu": 0.0,
                "mem_current": 0,
                "mem_peak": 0,
                "rss_peak_kb": 0,
                "config": self.config,
            }

        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        pop: List[List[int]] = []
        seen: Set[Tuple[int, ...]] = set()
        while len(pop) < self.config.population_size:
            cand = self._random_chromosome(n)
            key = canonical_tour(cand)
            if key not in seen:
                seen.add(key)
                pop.append(cand)

        fitness = [self._fitness(coords, ch) for ch in pop]
        best_idx = min(range(len(pop)), key=lambda i: fitness[i])
        best_tour = pop[best_idx][:]
        best_distance = fitness[best_idx]

        history: List[float] = []
        improvements: List[Tuple[int, float, List[int]]] = [(0, best_distance, best_tour[:])]

        for gen in range(1, self.config.generations + 1):
            new_pop: List[List[int]] = []

            elite_idx = min(range(len(pop)), key=lambda i: fitness[i])
            elite = pop[elite_idx][:]
            new_pop.append(elite)

            while len(new_pop) < self.config.population_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)

                if self.rng.random() < self.config.crossover_rate:
                    child = self._ox(p1, p2)
                else:
                    child = p1[:]

                self._mutate(child)

                if self.config.apply_2opt_on_new and (n <= 200 or self.rng.random() < self.config.p_ls_child):
                    child = two_opt(child, coords)

                new_pop.append(child)

            pop = new_pop
            fitness = [self._fitness(coords, ch) for ch in pop]

            gen_best_idx = min(range(len(pop)), key=lambda i: fitness[i])
            gen_best_tour = pop[gen_best_idx]
            gen_best_dist = fitness[gen_best_idx]

            if gen_best_dist < best_distance:
                key = canonical_tour(gen_best_tour)
                min_d = 1.0
                for other in pop:
                    if other is gen_best_tour:
                        continue
                    d = diversity_edges(gen_best_tour, other)
                    if d < min_d:
                        min_d = d
                if key not in seen and min_d + 1e-12 >= self.config.min_diversity:
                    seen.add(key)
                    best_distance = gen_best_dist
                    best_tour = gen_best_tour[:]
                    improvements.append((gen, best_distance, best_tour[:]))
                elif key in seen:
                    best_distance = gen_best_dist
                    best_tour = gen_best_tour[:]
                    improvements.append((gen, best_distance, best_tour[:]))

            history.append(best_distance)

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "best_tour": best_tour,
            "best_distance": best_distance,
            "history": history,
            "improvements": improvements,
            "time_wall": wall_end - wall_start,
            "time_cpu": cpu_end - cpu_start,
            "mem_current": mem_current,
            "mem_peak": mem_peak,
            "rss_peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "config": self.config,
        }


def _save_outputs(
    coords: List[Tuple[float, float]],
    result: Dict[str, Any],
    output_dir: str,
    output_prefix: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    best_tour = result["best_tour"]
    history = result["history"]

    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("CBGA (TSP) - Mejor ruta final")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    xs_coords = [c[0] for c in coords]
    ys_coords = [c[1] for c in coords]
    ax1.scatter(xs_coords, ys_coords)
    if best_tour:
        xs = [coords[i][0] for i in best_tour] + [coords[best_tour[0]][0]]
        ys = [coords[i][1] for i in best_tour] + [coords[best_tour[0]][1]]
        ax1.plot(xs, ys, linewidth=1)

    best_route_img = os.path.join(output_dir, f"{output_prefix}_best_route.png")
    fig1.savefig(best_route_img, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.set_title("CBGA (TSP) - Convergencia")
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Mejor distancia")
    ax2.plot(range(1, len(history) + 1), history, linewidth=1)
    conv_img = os.path.join(output_dir, f"{output_prefix}_convergence.png")
    fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    report_index = 1
    while True:
        candidate_report = os.path.join(output_dir, f"{output_prefix}_summary{report_index}.txt")
        if not os.path.exists(candidate_report):
            report_path = candidate_report
            break
        report_index += 1

    config: CBGAConfig = result["config"]
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("CBGA para TSP - Resumen de ejecución\n")
        rep.write("=" * 40 + "\n")
        rep.write(f"Nodos: {len(coords)}\n")
        rep.write(f"Población: {config.population_size}\n")
        rep.write(f"Generaciones: {config.generations}\n")
        rep.write(f"Crossover rate: {config.crossover_rate}\n")
        rep.write(f"Mutation rate: {config.mutation_rate}\n")
        rep.write(f"Tournament k: {config.tournament_k}\n")
        rep.write(f"Min diversity: {config.min_diversity}\n")
        rep.write(f"2-opt en hijos: {config.apply_2opt_on_new}\n")
        rep.write(f"Seed: {config.seed}\n\n")

        rep.write("Recursos consumidos\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"Tiempo de ejecución (wall): {result['time_wall']:.6f} s\n")
        rep.write(f"Tiempo de CPU: {result['time_cpu']:.6f} s\n")
        rep.write(f"Memoria actual (tracemalloc): {result['mem_current'] / (1024 * 1024):.3f} MiB\n")
        rep.write(f"Memoria pico (tracemalloc): {result['mem_peak'] / (1024 * 1024):.3f} MiB\n")
        rep.write(f"RSS pico del proceso: {result['rss_peak_kb']} KiB\n\n")

        rep.write("Resultado final\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"Mejor distancia final: {result['best_distance']:.6f}\n")
        rep.write(f"Mejor ruta final: {result['best_tour']}\n\n")

        rep.write("Mejores rutas a lo largo de la ejecución\n")
        rep.write("-" * 36 + "\n")
        for gen, dist, route in result["improvements"]:
            rep.write(f"gen={gen:4d} | dist={dist:.6f} | route={route}\n")

        rep.write("\nArchivos generados\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"{best_route_img}\n")
        rep.write(f"{conv_img}\n")
        rep.write(f"{report_path}\n")


def ejecutar_cbga_tsp(
    coords: List[Tuple[float, float]],
    config: Optional[CBGAConfig] = None,
    output_dir: str = "outputs",
    output_prefix: str = "cbga_tsp",
    save_outputs: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    if config is None:
        config = CBGAConfig()

    if save_outputs and not show_plot:
        plt.switch_backend("Agg")

    solver = CBGATSPSolver(config)
    result = solver.solve(coords)

    if save_outputs:
        _save_outputs(coords, result, output_dir, output_prefix)

    if show_plot and result["best_tour"]:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("CBGA (TSP) - Mejor ruta final")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter([c[0] for c in coords], [c[1] for c in coords])
        route = result["best_tour"]
        xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
        ys = [coords[i][1] for i in route] + [coords[route[0]][1]]
        ax.plot(xs, ys, linewidth=1)
        plt.show()

    return result


def cbga(
    coords: List[Tuple[float, float]],
    pop_size: int = 100,
    generations: int = 500,
    min_diversity: float = 0.2,
    p_mut: float = 0.2,
    p_ls_child: float = 0.1,
    seed: Optional[int] = None,
    apply_2opt_on_new: bool = True,
    tournament_k: int = 3,
) -> Tuple[List[int], float]:
    config = CBGAConfig(
        population_size=pop_size,
        generations=generations,
        mutation_rate=p_mut,
        tournament_k=tournament_k,
        min_diversity=min_diversity,
        p_ls_child=p_ls_child,
        apply_2opt_on_new=apply_2opt_on_new,
        seed=19 if seed is None else seed,
    )
    result = ejecutar_cbga_tsp(
        coords=coords,
        config=config,
        output_dir="outputs",
        output_prefix="cbga_tsp",
        save_outputs=True,
        show_plot=False,
    )
    return result["best_tour"], result["best_distance"]


def chu_beasley(coords: List[Tuple[float, float]], restarts: int = 10, seed: Optional[int] = None) -> Tuple[List[int], float]:
    rng = random.Random(seed)
    n = len(coords)
    if n == 0:
        return [], 0.0

    best_tour = None
    best_dist = float("inf")
    for _ in range(restarts):
        start = rng.randrange(n)
        tour = nearest_neighbor(coords, start=start)
        tour = two_opt(tour, coords)
        dist = total_distance(tour, coords)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour
    return best_tour if best_tour is not None else [], best_dist


def parse_args():
    p = argparse.ArgumentParser(description="Algoritmo Chu-Beasley (heurística TSP) y CBGA")
    p.add_argument("instance", nargs="?", default="berlin52.tsp", help="Archivo .tsp (EUC_2D)")
    p.add_argument("--restarts", type=int, default=20, help="Número de reinicios aleatorios (heurística)")

    p.add_argument("--ga", action="store_true", help="Ejecutar CBGA")
    p.add_argument("--pop", type=int, default=40, help="Tamaño de población para CBGA")
    p.add_argument("--generations", type=int, default=160, help="Generaciones para CBGA")
    p.add_argument("--cross-rate", type=float, default=0.9, help="Probabilidad de cruce")
    p.add_argument("--p-mut", type=float, default=0.25, help="Probabilidad de mutación swap")
    p.add_argument("--tournament-k", type=int, default=3, help="Tamaño del torneo")
    p.add_argument("--min-diversity", type=float, default=0.2, help="Umbral mínimo de diversidad")
    p.add_argument("--p-ls-child", type=float, default=0.1, help="Probabilidad de aplicar 2-opt a hijos")
    p.add_argument("--no-2opt", action="store_true", help="Desactiva 2-opt en hijos")
    p.add_argument("--seed", type=int, default=19, help="Semilla aleatoria")

    p.add_argument("--output-dir", default="outputs", help="Carpeta para salidas")
    p.add_argument("--output-prefix", default="cbga_tsp", help="Prefijo de archivos de salida")
    p.add_argument("--show-plot", action="store_true", help="Mostrar gráfico en pantalla")
    return p.parse_args()


def main():
    args = parse_args()
    coords = parse_tsp(args.instance)
    if not coords:
        print("No se encontraron coordenadas en la instancia.")
        sys.exit(1)

    if args.ga:
        config = CBGAConfig(
            population_size=args.pop,
            generations=args.generations,
            crossover_rate=args.cross_rate,
            mutation_rate=args.p_mut,
            tournament_k=args.tournament_k,
            min_diversity=args.min_diversity,
            p_ls_child=args.p_ls_child,
            apply_2opt_on_new=not args.no_2opt,
            seed=args.seed,
        )

        result = ejecutar_cbga_tsp(
            coords=coords,
            config=config,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            save_outputs=True,
            show_plot=args.show_plot,
        )
        dist = result["best_distance"]
        tour = result["best_tour"]
    else:
        tour, dist = chu_beasley(coords, restarts=args.restarts, seed=args.seed)

    print(f"Instancia: {args.instance}")
    print(f"Nodos: {len(coords)}")
    print(f"Distancia mejor tour: {dist:.4f}")
    print("Tour (0-based indices):")
    print(tour)


if __name__ == "__main__":
    main()
