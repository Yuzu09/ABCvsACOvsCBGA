"""ABC para VRP con salidas headless (imágenes + resumen de texto)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import os
import random
import time
import tracemalloc
import resource
import itertools

import numpy as np
import matplotlib.pyplot as plt

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


@dataclass
class ABCConfig:
	colony_size: int = 30
	limit: int = 30
	max_iters: int = 200
	neighborhood_trials: int = 3
	seed: int = 7
	seed_mode: str = "mixed"
	elite_local_search: bool = True


@dataclass
class FoodSource:
	solution: VRPSolution
	trials: int = 0


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
	return sum(route_cost(inst, route) for route in routes)


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
	seed: int = 7,
	capacity: int = 30,
	demand_low: int = 1,
	demand_high: int = 9,
) -> VRPInstance:
	rng = np.random.default_rng(seed)
	coords = rng.random((n_customers + 1, 2), dtype=float)
	demands = [0] + rng.integers(demand_low, demand_high + 1, size=n_customers).tolist()
	return VRPInstance(coords=coords, demands=demands, vehicle_capacity=capacity, depot=0)


def random_greedy_solution(inst: VRPInstance, rng: random.Random) -> VRPSolution:
	customers = list(range(len(inst.coords)))
	customers.remove(inst.depot)
	rng.shuffle(customers)

	routes: List[List[int]] = []
	current_route: List[int] = []
	current_load = 0
	for customer in customers:
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

	return VRPSolution(routes=routes, cost=solution_cost(inst, routes))


def nearest_neighbor_seed(inst: VRPInstance, rng: random.Random) -> VRPSolution:
	unvisited = set(range(len(inst.coords)))
	unvisited.remove(inst.depot)
	routes: List[List[int]] = []

	while unvisited:
		route: List[int] = []
		load = 0
		current = inst.depot

		while True:
			feasible = [c for c in unvisited if load + inst.demands[c] <= inst.vehicle_capacity]
			if not feasible:
				break
			next_customer = min(feasible, key=lambda c: euclidean(inst.coords[current], inst.coords[c]))
			route.append(next_customer)
			unvisited.remove(next_customer)
			load += inst.demands[next_customer]
			current = next_customer

		if not route:
			single = rng.choice(list(unvisited))
			route = [single]
			unvisited.remove(single)

		routes.append(route)

	return VRPSolution(routes=routes, cost=solution_cost(inst, routes))


def perturb(inst: VRPInstance, sol: VRPSolution, rng: random.Random) -> VRPSolution:
	routes = [r[:] for r in sol.routes]
	if not routes:
		return sol.copy()

	op = rng.choice(["relocate", "swap", "reverse", "cross_exchange"])

	if op == "reverse":
		non_empty = [r for r in routes if len(r) >= 2]
		if non_empty:
			route = rng.choice(non_empty)
			i, j = sorted(rng.sample(range(len(route)), 2))
			route[i : j + 1] = reversed(route[i : j + 1])

	elif op == "swap":
		non_empty_idx = [i for i, r in enumerate(routes) if r]
		if len(non_empty_idx) >= 2:
			a_idx, b_idx = rng.sample(non_empty_idx, 2)
			a_pos = rng.randrange(len(routes[a_idx]))
			b_pos = rng.randrange(len(routes[b_idx]))
			a_c = routes[a_idx][a_pos]
			b_c = routes[b_idx][b_pos]

			new_a = route_demand(inst, routes[a_idx]) - inst.demands[a_c] + inst.demands[b_c]
			new_b = route_demand(inst, routes[b_idx]) - inst.demands[b_c] + inst.demands[a_c]
			if new_a <= inst.vehicle_capacity and new_b <= inst.vehicle_capacity:
				routes[a_idx][a_pos], routes[b_idx][b_pos] = routes[b_idx][b_pos], routes[a_idx][a_pos]

	elif op == "cross_exchange":
		non_empty_idx = [i for i, r in enumerate(routes) if r]
		if len(non_empty_idx) >= 2:
			a_idx, b_idx = rng.sample(non_empty_idx, 2)
			route_a = routes[a_idx]
			route_b = routes[b_idx]

			len_a = rng.randint(1, min(2, len(route_a)))
			len_b = rng.randint(1, min(2, len(route_b)))

			start_a = rng.randint(0, len(route_a) - len_a)
			start_b = rng.randint(0, len(route_b) - len_b)

			seg_a = route_a[start_a : start_a + len_a]
			seg_b = route_b[start_b : start_b + len_b]

			new_route_a = route_a[:start_a] + seg_b + route_a[start_a + len_a :]
			new_route_b = route_b[:start_b] + seg_a + route_b[start_b + len_b :]

			if route_demand(inst, new_route_a) <= inst.vehicle_capacity and route_demand(inst, new_route_b) <= inst.vehicle_capacity:
				routes[a_idx] = new_route_a
				routes[b_idx] = new_route_b

	else:
		non_empty_idx = [i for i, r in enumerate(routes) if r]
		if non_empty_idx:
			from_idx = rng.choice(non_empty_idx)
			from_route = routes[from_idx]
			pos = rng.randrange(len(from_route))
			customer = from_route.pop(pos)
			target_idx = rng.randrange(len(routes))
			target_route = routes[target_idx]

			if route_demand(inst, target_route) + inst.demands[customer] <= inst.vehicle_capacity:
				insert_pos = rng.randrange(len(target_route) + 1)
				target_route.insert(insert_pos, customer)
			else:
				from_route.insert(pos, customer)

	routes = [r for r in routes if r]
	if not routes:
		return sol.copy()

	candidate = VRPSolution(routes=routes, cost=solution_cost(inst, routes))
	if not is_feasible(inst, candidate):
		return sol.copy()
	return candidate


class ABCVRPSolver:
	def __init__(self, config: ABCConfig):
		self.config = config
		self.rng = random.Random(config.seed)

	def _make_solution(self, inst: VRPInstance) -> VRPSolution:
		mode = self.config.seed_mode
		if mode == "random":
			return random_greedy_solution(inst, self.rng)
		if mode == "nn":
			return nearest_neighbor_seed(inst, self.rng)
		return self.rng.choice([
			lambda: random_greedy_solution(inst, self.rng),
			lambda: nearest_neighbor_seed(inst, self.rng),
		])()

	def _init_sources(self, inst: VRPInstance) -> List[FoodSource]:
		n_sources = max(2, self.config.colony_size // 2)
		return [FoodSource(self._make_solution(inst)) for _ in range(n_sources)]

	def _neighbor(self, inst: VRPInstance, sol: VRPSolution) -> VRPSolution:
		best = sol.copy()
		for _ in range(self.config.neighborhood_trials):
			cand = perturb(inst, sol, self.rng)
			if cand.cost < best.cost:
				best = cand
		return best

	@staticmethod
	def _fitness(cost: float) -> float:
		return 1.0 / (1.0 + cost)

	def solve(self, inst: VRPInstance) -> Dict[str, Any]:
		tracemalloc.start()
		wall_start = time.perf_counter()
		cpu_start = time.process_time()

		sources = self._init_sources(inst)
		best = min((fs.solution for fs in sources), key=lambda s: s.cost).copy()
		history: List[float] = []
		improvements: List[Tuple[int, float, List[List[int]]]] = [(0, best.cost, [r[:] for r in best.routes])]

		for iteration in range(1, self.config.max_iters + 1):
			for fs in sources:
				cand = self._neighbor(inst, fs.solution)
				if cand.cost < fs.solution.cost:
					fs.solution = cand
					fs.trials = 0
				else:
					fs.trials += 1

			fits = [self._fitness(fs.solution.cost) for fs in sources]
			total_fit = sum(fits)
			if total_fit <= 0.0:
				probs = [1.0 / len(sources)] * len(sources)
			else:
				probs = [f / total_fit for f in fits]

			for _ in range(len(sources)):
				idx = self.rng.choices(range(len(sources)), weights=probs, k=1)[0]
				fs = sources[idx]
				cand = self._neighbor(inst, fs.solution)
				if cand.cost < fs.solution.cost:
					fs.solution = cand
					fs.trials = 0
				else:
					fs.trials += 1

			for fs in sources:
				if fs.trials >= self.config.limit:
					fs.solution = self._make_solution(inst)
					fs.trials = 0

			current_best = min((fs.solution for fs in sources), key=lambda s: s.cost)
			if current_best.cost < best.cost:
				best = current_best.copy()
				if self.config.elite_local_search:
					for _ in range(5):
						improved = self._neighbor(inst, best)
						if improved.cost < best.cost:
							best = improved
				improvements.append((iteration, best.cost, [r[:] for r in best.routes]))

			history.append(best.cost)

		wall_end = time.perf_counter()
		cpu_end = time.process_time()
		mem_current, mem_peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()

		elapsed = wall_end - wall_start
		return {
			"best_solution": best,
			"best_cost": best.cost,
			"time_wall": elapsed,
			"time_cpu": cpu_end - cpu_start,
			"history": history,
			"feasible": is_feasible(inst, best),
			"config": self.config,
			"improvements": improvements,
			"mem_current": mem_current,
			"mem_peak": mem_peak,
			"rss_peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
		}


def _next_indexed_path(output_dir: str, output_prefix: str, base_name: str, extension: str) -> str:
	idx = 1
	while True:
		candidate = os.path.join(output_dir, f"{output_prefix}_{base_name}{idx}.{extension}")
		if not os.path.exists(candidate):
			return candidate
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

	fig1, ax1 = plt.subplots()
	ax1.set_aspect("equal", adjustable="box")
	ax1.set_title("ABC (VRP) - Mejor solución final")
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

	fig2, ax2 = plt.subplots()
	ax2.set_title("ABC (VRP) - Convergencia")
	ax2.set_xlabel("Iteración")
	ax2.set_ylabel("Mejor costo")
	ax2.plot(range(1, len(history) + 1), history, linewidth=1)
	conv_img = _next_indexed_path(output_dir, output_prefix, "convergence", "png")
	fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
	plt.close(fig2)

	report_path = _next_indexed_path(output_dir, output_prefix, "summary", "txt")
	config: ABCConfig = result["config"]
	with open(report_path, "w", encoding="utf-8") as rep:
		rep.write("ABC para VRP - Resumen de ejecución\n")
		rep.write("=" * 40 + "\n")
		rep.write(f"Clientes: {len(inst.coords) - 1}\n")
		rep.write(f"Capacidad vehículo: {inst.vehicle_capacity}\n")
		rep.write(f"Colony size: {config.colony_size}\n")
		rep.write(f"Limit: {config.limit}\n")
		rep.write(f"Max iters: {config.max_iters}\n")
		rep.write(f"Neighborhood trials: {config.neighborhood_trials}\n")
		rep.write(f"Seed mode: {config.seed_mode}\n")
		rep.write(f"Elite local search: {config.elite_local_search}\n")
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


def run_abc_vrp(
	n_customers: int = 40,
	capacity: int = 30,
	seed: int = 7,
	config: Optional[ABCConfig] = None,
	output_dir: str = "outputs",
	output_prefix: str = "abc_vrp",
	save_outputs: bool = True,
	show_plot: bool = False,
) -> Dict[str, Any]:
	if config is None:
		config = ABCConfig(seed=seed)

	if save_outputs and not show_plot:
		plt.switch_backend("Agg")

	inst = build_random_vrp_instance(n_customers=n_customers, seed=seed, capacity=capacity)
	solver = ABCVRPSolver(config)
	result = solver.solve(inst)

	if save_outputs:
		_save_outputs(
			inst=inst,
			result=result,
			output_dir=output_dir,
			output_prefix=output_prefix,
		)

	if show_plot:
		best: VRPSolution = result["best_solution"]
		fig, ax = plt.subplots()
		ax.set_aspect("equal", adjustable="box")
		ax.set_title("ABC (VRP) - Mejor solución final")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1], s=15)
		ax.scatter(inst.coords[inst.depot, 0], inst.coords[inst.depot, 1], c="red", s=40)
		for route in best.routes:
			xs = [inst.coords[inst.depot, 0]] + [inst.coords[c, 0] for c in route] + [inst.coords[inst.depot, 0]]
			ys = [inst.coords[inst.depot, 1]] + [inst.coords[c, 1] for c in route] + [inst.coords[inst.depot, 1]]
			ax.plot(xs, ys, linewidth=1)
		plt.show()

	return result


if __name__ == "__main__":
	run_abc_vrp(
		n_customers=100,
		capacity=35,
		seed=2,
		config=ABCConfig(
			colony_size=20,
			limit=40,
			max_iters=500,
			neighborhood_trials=4,
			seed=4,
			seed_mode="nn",
			elite_local_search=True,
		),
		output_dir="outputs",
		output_prefix="abc_vrp",
		save_outputs=True,
		show_plot=False,
	)
