from __future__ import annotations

import math
import os
import random
import psutil
import time
import tracemalloc
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------
# Estructuras VRP
# ------------------------------------

@dataclass
class VRPInstance:
    coords: np.ndarray
    demands: List[int]
    vehicle_capacity: int
    depot: int = 0


@dataclass
class VRPSolution:
    routes: List[List[int]]
    cost: float

    def copy(self) -> "VRPSolution":
        return VRPSolution(routes=[r[:] for r in self.routes], cost=self.cost)


def euclidean_np(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def route_cost(inst: VRPInstance, route: List[int]) -> float:
    if not route:
        return 0.0
    cost = 0.0
    prev = inst.depot
    for customer in route:
        cost += euclidean_np(inst.coords[prev], inst.coords[customer])
        prev = customer
    cost += euclidean_np(inst.coords[prev], inst.coords[inst.depot])
    return cost


def solution_cost(inst: VRPInstance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst, r) for r in routes)


def route_demand(inst: VRPInstance, route: List[int]) -> int:
    return sum(inst.demands[c] for c in route)


def is_feasible(inst: VRPInstance, sol: VRPSolution) -> bool:
    customers = list(range(len(inst.coords)))
    customers.remove(inst.depot)
    seen: List[int] = list(itertools.chain.from_iterable(sol.routes))
    if sorted(seen) != customers:
        return False
    if len(set(seen)) != len(seen):
        return False
    for route in sol.routes:
        if any(c == inst.depot for c in route):
            return False
        if route_demand(inst, route) > inst.vehicle_capacity:
            return False
    return True


def build_random_vrp_instance(
    n_customers: int = 40,
    seed: int = 19,
    capacity: int = 30,
    demand_low: int = 1,
    demand_high: int = 9,
) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_customers + 1, 2), dtype=float)
    demands = [0] + rng.integers(demand_low, demand_high + 1, size=n_customers).tolist()
    return VRPInstance(coords=coords, demands=demands, vehicle_capacity=capacity, depot=0)


# ------------------------------------
# Giant-tour: codificación y decodificación
# ------------------------------------

def split_giant_tour(inst: VRPInstance, giant_tour: List[int]) -> List[List[int]]:
    """Divide la permutación de clientes en rutas factibles por capacidad."""
    routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0
    for customer in giant_tour:
        demand = inst.demands[customer]
        if current_load + demand <= inst.vehicle_capacity:
            current_route.append(customer)
            current_load += demand
        else:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_load = demand
    if current_route:
        routes.append(current_route)
    return routes


def giant_tour_cost(inst: VRPInstance, giant_tour: List[int]) -> float:
    routes = split_giant_tour(inst, giant_tour)
    return solution_cost(inst, routes)


def giant_tour_to_solution(inst: VRPInstance, giant_tour: List[int]) -> VRPSolution:
    routes = split_giant_tour(inst, giant_tour)
    return VRPSolution(routes=routes, cost=solution_cost(inst, routes))


# ------------------------------------
# Diversidad sobre permutación de clientes
# ------------------------------------

def diversity_permutation(a: List[int], b: List[int]) -> float:
    """Fracción de posiciones distintas entre dos permutaciones del mismo largo."""
    if not a:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    return diff / len(a)


# ------------------------------------
# CBGA para VRP
# ------------------------------------

@dataclass
class CBGAConfig:
    population_size: int = 40
    generations: int = 160
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    tournament_k: int = 3
    min_diversity: float = 0.15
    p_ls_child: float = 0.1
    apply_2opt_giant: bool = True   # 2-opt ligero sobre el giant-tour
    seed: int = 19


class CBGAVRPSolver:
    def __init__(self, config: CBGAConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def _random_chromosome(self, customers: List[int]) -> List[int]:
        chrom = customers[:]
        self.rng.shuffle(chrom)
        return chrom

    def _fitness(self, inst: VRPInstance, chrom: List[int]) -> float:
        return giant_tour_cost(inst, chrom)

    def _tournament(self, pop: List[List[int]], fitness: List[float]) -> List[int]:
        idxs = self.rng.sample(range(len(pop)), self.config.tournament_k)
        best_idx = min(idxs, key=lambda i: fitness[i])
        return pop[best_idx][:]

    def _ox(self, p1: List[int], p2: List[int]) -> List[int]:
        """Order Crossover (OX) sobre permutación de clientes."""
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
        """Mutación: swap de dos clientes en el giant-tour."""
        if self.rng.random() < self.config.mutation_rate:
            i, j = self.rng.sample(range(len(chrom)), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]

    def _local_search_2opt(self, inst: VRPInstance, chrom: List[int], max_tries: int = 40) -> List[int]:
        """2-opt ligero sobre el giant-tour: intenta inversiones aleatorias."""
        best = chrom[:]
        best_cost = giant_tour_cost(inst, best)
        n = len(best)
        for _ in range(max_tries):
            i, j = sorted(self.rng.sample(range(n), 2))
            cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
            c = giant_tour_cost(inst, cand)
            if c < best_cost:
                best, best_cost = cand, c
        return best

    def solve(self, inst: VRPInstance) -> Dict[str, Any]:
        customers = list(range(len(inst.coords)))
        customers.remove(inst.depot)
        n = len(customers)

        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        pop: List[List[int]] = []
        seen: Set[Tuple[int, ...]] = set()
        while len(pop) < self.config.population_size:
            cand = self._random_chromosome(customers)
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                pop.append(cand)

        fitness = [self._fitness(inst, ch) for ch in pop]
        best_idx = min(range(len(pop)), key=lambda i: fitness[i])
        best_giant = pop[best_idx][:]
        best_distance = fitness[best_idx]
        best_solution = giant_tour_to_solution(inst, best_giant)

        history: List[float] = []
        improvements: List[Tuple[int, float, List[List[int]]]] = [
            (0, best_distance, [r[:] for r in best_solution.routes])
        ]

        for gen in range(1, self.config.generations + 1):
            new_pop: List[List[int]] = []

            # Elitismo: el mejor pasa sin cambios
            elite_idx = min(range(len(pop)), key=lambda i: fitness[i])
            new_pop.append(pop[elite_idx][:])

            while len(new_pop) < self.config.population_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)

                if self.rng.random() < self.config.crossover_rate:
                    child = self._ox(p1, p2)
                else:
                    child = p1[:]

                self._mutate(child)

                if self.config.apply_2opt_giant and self.rng.random() < self.config.p_ls_child:
                    child = self._local_search_2opt(inst, child)

                new_pop.append(child)

            pop = new_pop
            fitness = [self._fitness(inst, ch) for ch in pop]

            gen_best_idx = min(range(len(pop)), key=lambda i: fitness[i])
            gen_best_giant = pop[gen_best_idx]
            gen_best_dist = fitness[gen_best_idx]

            if gen_best_dist < best_distance:
                # Criterio de diversidad Chu-Beasley
                min_d = 1.0
                for other in pop:
                    if other is gen_best_giant:
                        continue
                    d = diversity_permutation(gen_best_giant, other)
                    if d < min_d:
                        min_d = d
                if min_d + 1e-12 >= self.config.min_diversity:
                    best_distance = gen_best_dist
                    best_giant = gen_best_giant[:]
                    best_solution = giant_tour_to_solution(inst, best_giant)
                    improvements.append((gen, best_distance, [r[:] for r in best_solution.routes]))
                else:
                    # Acepta igual si ya fue visto (elitismo puro)
                    best_distance = gen_best_dist
                    best_giant = gen_best_giant[:]
                    best_solution = giant_tour_to_solution(inst, best_giant)
                    improvements.append((gen, best_distance, [r[:] for r in best_solution.routes]))

            history.append(best_distance)

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "best_solution": best_solution,
            "best_cost": best_distance,
            "history": history,
            "improvements": improvements,
            "time_wall": wall_end - wall_start,
            "time_cpu": cpu_end - cpu_start,
            "mem_current": mem_current,
            "mem_peak": mem_peak,
            "rss_peak_kb": psutil.Process().memory_info().rss / 1024,
            "config": self.config,
            "feasible": is_feasible(inst, best_solution),
        }


def _next_indexed_path(output_dir: str, prefix: str, base: str, ext: str) -> str:
    idx = 1
    while True:
        p = os.path.join(output_dir, f"{prefix}_{base}{idx}.{ext}")
        if not os.path.exists(p):
            return p
        idx += 1


def _save_outputs(
    inst: VRPInstance,
    result: Dict[str, Any],
    output_dir: str,
    output_prefix: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    best: VRPSolution = result["best_solution"]
    history: List[float] = result["history"]

    # Imagen 1: rutas VRP
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("CBGA (VRP) - Mejor solución final")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(inst.coords[1:, 0], inst.coords[1:, 1], label="Clientes", s=15)
    ax1.scatter(inst.coords[inst.depot, 0], inst.coords[inst.depot, 1], c="red", label="Depósito", s=40)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(best.routes))))
    for ridx, route in enumerate(best.routes):
        if not route:
            continue
        xs = [inst.coords[inst.depot, 0]] + [inst.coords[c, 0] for c in route] + [inst.coords[inst.depot, 0]]
        ys = [inst.coords[inst.depot, 1]] + [inst.coords[c, 1] for c in route] + [inst.coords[inst.depot, 1]]
        ax1.plot(xs, ys, linewidth=1.2, color=colors[ridx])
    ax1.legend(loc="best", fontsize=8)
    best_route_img = _next_indexed_path(output_dir, output_prefix, "best_routes", "png")
    fig1.savefig(best_route_img, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    # Imagen 2: convergencia (incremental)
    fig2, ax2 = plt.subplots()
    ax2.set_title("CBGA (VRP) - Convergencia")
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Mejor costo")
    ax2.plot(range(1, len(history) + 1), history, linewidth=1)
    conv_img = _next_indexed_path(output_dir, output_prefix, "convergence", "png")
    fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Reporte incremental
    report_path = _next_indexed_path(output_dir, output_prefix, "summary", "txt")
    config: CBGAConfig = result["config"]
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("CBGA para VRP - Resumen de ejecución\n")
        rep.write("=" * 40 + "\n")
        rep.write(f"Clientes: {len(inst.coords) - 1}\n")
        rep.write(f"Capacidad vehículo: {inst.vehicle_capacity}\n")
        rep.write(f"Población: {config.population_size}\n")
        rep.write(f"Generaciones: {config.generations}\n")
        rep.write(f"Crossover rate: {config.crossover_rate}\n")
        rep.write(f"Mutation rate: {config.mutation_rate}\n")
        rep.write(f"Tournament k: {config.tournament_k}\n")
        rep.write(f"Min diversity: {config.min_diversity}\n")
        rep.write(f"2-opt giant-tour: {config.apply_2opt_giant}\n")
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
        rep.write(f"Mejor costo final: {result['best_cost']:.6f}\n")
        rep.write(f"Factible: {result['feasible']}\n")
        rep.write(f"Número de rutas: {len(best.routes)}\n")
        rep.write(f"Rutas finales: {best.routes}\n\n")

        rep.write("Mejoras a lo largo de la ejecución\n")
        rep.write("-" * 36 + "\n")
        for gen, cost, routes in result["improvements"]:
            rep.write(f"gen={gen:4d} | cost={cost:.6f} | routes={routes}\n")

        rep.write("\nArchivos generados\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"{best_route_img}\n")
        rep.write(f"{conv_img}\n")
        rep.write(f"{report_path}\n")


def run_cbga_vrp(
    n_customers: int = 60,
    capacity: int = 35,
    seed: int = 19,
    config: Optional[CBGAConfig] = None,
    output_dir: str = "outputs",
    output_prefix: str = "cbga_vrp",
    save_outputs: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    if config is None:
        config = CBGAConfig(seed=seed)

    if save_outputs and not show_plot:
        plt.switch_backend("Agg")

    inst = build_random_vrp_instance(n_customers=n_customers, seed=seed, capacity=capacity)
    solver = CBGAVRPSolver(config)
    result = solver.solve(inst)

    if save_outputs:
        _save_outputs(inst, result, output_dir, output_prefix)

    if show_plot:
        best: VRPSolution = result["best_solution"]
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("CBGA (VRP) - Mejor solución final")
        ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1], s=15)
        ax.scatter(inst.coords[inst.depot, 0], inst.coords[inst.depot, 1], c="red", s=40)
        for route in best.routes:
            xs = [inst.coords[inst.depot, 0]] + [inst.coords[c, 0] for c in route] + [inst.coords[inst.depot, 0]]
            ys = [inst.coords[inst.depot, 1]] + [inst.coords[c, 1] for c in route] + [inst.coords[inst.depot, 1]]
            ax.plot(xs, ys, linewidth=1)
        plt.show()

    return result


if __name__ == "__main__":
    run_cbga_vrp(
        n_customers=100,
        capacity=35,
        seed=2,
        config=CBGAConfig(
            population_size=40,
            generations=160,
            crossover_rate=0.9,
            mutation_rate=0.25,
            tournament_k=3,
            min_diversity=0.15,
            p_ls_child=0.1,
            apply_2opt_giant=True,
            seed=19,
        ),
        output_dir="outputs",
        output_prefix="cbga_vrp",
        save_outputs=True,
        show_plot=False,
    )
