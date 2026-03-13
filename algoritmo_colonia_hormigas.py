"""ACO para VRP con salidas headless (imágenes + resumen de texto)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import os
import random
import time
import tracemalloc
import resource
import itertools

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


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def route_cost(inst: VRPInstance, route: List[int]) -> float:
    if not route:
        return 0.0
    cost = 0.0
    prev = inst.depot
    for customer in route:
        cost += euclidean(inst.coords[prev], inst.coords[customer])
        prev = customer
    cost += euclidean(inst.coords[prev], inst.coords[inst.depot])
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
    seed: int = 11,
    capacity: int = 30,
    demand_low: int = 1,
    demand_high: int = 9,
) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_customers + 1, 2), dtype=float)
    demands = [0] + rng.integers(demand_low, demand_high + 1, size=n_customers).tolist()
    return VRPInstance(coords=coords, demands=demands, vehicle_capacity=capacity, depot=0)


# ------------------------------------
# ACO para VRP
# ------------------------------------

@dataclass
class ACOConfig:
    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.3
    q: float = 100.0
    max_iters: int = 300
    feromona_inicial: float = 1.0
    seed: int = 11


class ACOVRPSolver:
    def __init__(self, config: ACOConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def _build_solution(self, inst: VRPInstance, tau: np.ndarray) -> VRPSolution:
        remaining = set(range(len(inst.coords)))
        remaining.remove(inst.depot)
        routes: List[List[int]] = []

        while remaining:
            route: List[int] = []
            current = inst.depot
            load = 0

            while True:
                feasible = [c for c in remaining if load + inst.demands[c] <= inst.vehicle_capacity]
                if not feasible:
                    break

                desirabilities = []
                for c in feasible:
                    pher = tau[current, c] ** self.config.alpha
                    dist = euclidean(inst.coords[current], inst.coords[c])
                    heur = (1.0 / (dist + 1e-9)) ** self.config.beta
                    desirabilities.append(pher * heur)

                total = sum(desirabilities)
                if total <= 0:
                    chosen = self.rng.choice(feasible)
                else:
                    probs = [d / total for d in desirabilities]
                    chosen = self.rng.choices(feasible, weights=probs, k=1)[0]

                route.append(chosen)
                remaining.remove(chosen)
                load += inst.demands[chosen]
                current = chosen

            # Si ningún cliente cupió (demanda > capacidad), forzar uno solo
            if not route and remaining:
                single = next(iter(remaining))
                route = [single]
                remaining.remove(single)

            if route:
                routes.append(route)

        return VRPSolution(routes=routes, cost=solution_cost(inst, routes))

    def solve(self, inst: VRPInstance) -> Dict[str, Any]:
        n = len(inst.coords)
        tau = np.ones((n, n), dtype=float) * self.config.feromona_inicial

        best: Optional[VRPSolution] = None
        history: List[float] = []
        improvements: List[Tuple[int, float, List[List[int]]]] = []

        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        for iteration in range(1, self.config.max_iters + 1):
            ants = [self._build_solution(inst, tau) for _ in range(self.config.n_ants)]
            iter_best = min(ants, key=lambda s: s.cost)

            if best is None or iter_best.cost < best.cost:
                best = iter_best.copy()
                improvements.append((iteration, best.cost, [r[:] for r in best.routes]))

            # Evaporación
            tau *= (1.0 - self.config.evaporation)

            # Depósito de feromona
            for sol in ants:
                if sol.cost <= 0:
                    continue
                deposit = self.config.q / sol.cost
                for route in sol.routes:
                    prev = inst.depot
                    for c in route:
                        tau[prev, c] += deposit
                        tau[c, prev] += deposit
                        prev = c
                    tau[prev, inst.depot] += deposit
                    tau[inst.depot, prev] += deposit

            history.append(best.cost)

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if not improvements and best is not None:
            improvements = [(0, best.cost, [r[:] for r in best.routes])]

        return {
            "best_solution": best,
            "best_cost": best.cost if best is not None else float("inf"),
            "history": history,
            "improvements": improvements,
            "time_wall": wall_end - wall_start,
            "time_cpu": cpu_end - cpu_start,
            "mem_current": mem_current,
            "mem_peak": mem_peak,
            "rss_peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "feasible": is_feasible(inst, best) if best is not None else False,
            "config": self.config,
        }


# ------------------------------------
# Salidas
# ------------------------------------

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

    # Imagen 1: rutas VRP coloreadas
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("ACO (VRP) - Mejor solución final")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(inst.coords[1:, 0], inst.coords[1:, 1], label="Clientes", s=15)
    ax1.scatter(inst.coords[inst.depot, 0], inst.coords[inst.depot, 1],
                c="red", zorder=5, label="Depósito", s=40)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(best.routes))))
    for ridx, route in enumerate(best.routes):
        if not route:
            continue
        xs = ([inst.coords[inst.depot, 0]]
              + [inst.coords[c, 0] for c in route]
              + [inst.coords[inst.depot, 0]])
        ys = ([inst.coords[inst.depot, 1]]
              + [inst.coords[c, 1] for c in route]
              + [inst.coords[inst.depot, 1]])
        ax1.plot(xs, ys, linewidth=1.2, color=colors[ridx])
    ax1.legend(loc="best", fontsize=8)
    best_route_img = _next_indexed_path(output_dir, output_prefix, "best_routes", "png")
    fig1.savefig(best_route_img, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    # Imagen 2: convergencia (incremental)
    fig2, ax2 = plt.subplots()
    ax2.set_title("ACO (VRP) - Convergencia")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Mejor costo")
    ax2.plot(range(1, len(history) + 1), history, linewidth=1)
    conv_img = _next_indexed_path(output_dir, output_prefix, "convergence", "png")
    fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Reporte incremental
    report_path = _next_indexed_path(output_dir, output_prefix, "summary", "txt")
    config: ACOConfig = result["config"]
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("ACO para VRP - Resumen de ejecución\n")
        rep.write("=" * 40 + "\n")
        rep.write(f"Clientes: {len(inst.coords) - 1}\n")
        rep.write(f"Capacidad vehículo: {inst.vehicle_capacity}\n")
        rep.write(f"Hormigas: {config.n_ants}\n")
        rep.write(f"Alpha: {config.alpha}\n")
        rep.write(f"Beta: {config.beta}\n")
        rep.write(f"Evaporación: {config.evaporation}\n")
        rep.write(f"Q: {config.q}\n")
        rep.write(f"Max iters: {config.max_iters}\n")
        rep.write(f"Feromona inicial: {config.feromona_inicial}\n")
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
        for iteration, cost, routes in result["improvements"]:
            rep.write(f"iter={iteration:4d} | cost={cost:.6f} | routes={routes}\n")

        rep.write("\nArchivos generados\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"{best_route_img}\n")
        rep.write(f"{conv_img}\n")
        rep.write(f"{report_path}\n")


def run_aco_vrp(
    n_customers: int = 60,
    capacity: int = 35,
    seed: int = 11,
    config: Optional[ACOConfig] = None,
    output_dir: str = "outputs",
    output_prefix: str = "aco_vrp",
    save_outputs: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    if config is None:
        config = ACOConfig(seed=seed)

    if save_outputs and not show_plot:
        plt.switch_backend("Agg")

    inst = build_random_vrp_instance(n_customers=n_customers, seed=seed, capacity=capacity)
    solver = ACOVRPSolver(config)
    result = solver.solve(inst)

    if save_outputs:
        _save_outputs(inst, result, output_dir, output_prefix)

    if show_plot and result["best_solution"] is not None:
        best: VRPSolution = result["best_solution"]
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("ACO (VRP) - Mejor solución final")
        ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1], s=15)
        ax.scatter(inst.coords[inst.depot, 0], inst.coords[inst.depot, 1], c="red", s=40)
        for route in best.routes:
            xs = ([inst.coords[inst.depot, 0]]
                  + [inst.coords[c, 0] for c in route]
                  + [inst.coords[inst.depot, 0]])
            ys = ([inst.coords[inst.depot, 1]]
                  + [inst.coords[c, 1] for c in route]
                  + [inst.coords[inst.depot, 1]])
            ax.plot(xs, ys, linewidth=1)
        plt.show()

    return result


if __name__ == "__main__":
    run_aco_vrp(
        n_customers=60,
        capacity=35,
        seed=11,
        config=ACOConfig(
            n_ants=5,
            alpha=1.0,
            beta=5.0,
            evaporation=0.3,
            q=100.0,
            max_iters=200,
            feromona_inicial=1.0,
            seed=11,
        ),
        output_dir="outputs",
        output_prefix="aco_vrp",
        save_outputs=True,
        show_plot=False,
    )
