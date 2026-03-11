"""
Animación ABC (versión didáctica) para TSP en Matplotlib
-----------------------------------------------------
- Enfocado a docencia: muestra cómo mejora el tour a lo largo de ciclos.
- Implementa ABC discreto: vecindarios swap y 2-opt (opcional).
- NO requiere librerías raras: solo numpy y matplotlib.

Uso:
  python abc_tsp_anim.py

En Jupyter:
  %run abc_tsp_anim.py

Autor: SoftComputer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import os
import random
import time
import tracemalloc
import resource

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Datos y utilidades de TSP
# ----------------------------
def make_cities(n: int = 30, seed: int = 0) -> np.ndarray:
	rng = np.random.default_rng(seed)
	# puntos en [0,1]x[0,1]
	return rng.random((n, 2), dtype=float)
	
def tour_length(cities: np.ndarray, tour: List[int]) -> float:
	# distancia euclídea del ciclo
	d = 0.0
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i + 1) % len(tour)]
		dx = cities[a, 0] - cities[b, 0]
		dy = cities[a, 1] - cities[b, 1]
		d += math.hypot(dx, dy)
	return d
	
def random_tour(n: int) -> List[int]:
	tour = list(range(n))
	random.shuffle(tour)
	return tour
	
def fitness_from_cost(cost: float) -> float:
	# minimización -> maximización
	return 1.0 / (1.0 + cost)
	
def clamp01(x: float) -> float:
	return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
	
	
# ----------------------------
# Vecindarios discretos
# ----------------------------
def move_swap(tour: List[int]) -> List[int]:
	n = len(tour)
	i, j = random.sample(range(n), 2)
	new = tour[:]
	new[i], new[j] = new[j], new[i]
	return new
	
def move_insert(tour: List[int]) -> List[int]:
	n = len(tour)
	i, j = random.sample(range(n), 2)
	new = tour[:]
	city = new.pop(i)
	new.insert(j, city)
	return new
	
def move_2opt(tour: List[int]) -> List[int]:
	n = len(tour)
	i, j = sorted(random.sample(range(n), 2))
	if i == j:
		return tour[:]
	new = tour[:]
	new[i:j+1] = reversed(new[i:j+1])
	return new
	
def local_2opt_best_improvement(cities: np.ndarray, tour: List[int], max_tries: int = 60) -> List[int]:
	"""2-opt ligero (didáctico): intenta algunas inversiones aleatorias y se queda con la mejor."""
	best = tour[:]
	best_cost = tour_length(cities, best)
	for _ in range(max_tries):
		cand = move_2opt(best)
		c = tour_length(cities, cand)
		if c < best_cost:
			best, best_cost = cand, c
	return best
	
	
# ----------------------------
# ABC discreto (TSP)
# ----------------------------
@dataclass
class FoodSource:
	tour: List[int]
	cost: float
	fitness: float
	trial: int = 0
	
@dataclass
class ABCParams:
	SN: int = 40               # número de fuentes
	max_cycle: int = 400       # iteraciones
	limit: int = 120           # abandono
	onlookers: Optional[int] = None  # por defecto = SN
	move: str = "swap"         # "swap" | "insert" | "2opt"
	intensify_2opt: bool = True       # refina mejor global con 2-opt ligero
	seed: int = 0
	
class ABC_TSP:
	def __init__(self, cities: np.ndarray, params: ABCParams):
		self.cities = cities
		self.p = params
		random.seed(self.p.seed)
		np.random.seed(self.p.seed)
		
		self.n = cities.shape[0]
		self.onlookers = self.p.onlookers if self.p.onlookers is not None else self.p.SN
		
		self.sources: List[FoodSource] = []
		self.best: Optional[FoodSource] = None
		self.history_best_cost: List[float] = []
		self.history_best_tour: List[List[int]] = []
		self.improvements: List[Tuple[int, float, List[int]]] = []
		self.cycle: int = 0
		
		self._init_sources()
		
	def _init_sources(self):
		self.sources.clear()
		for _ in range(self.p.SN):
			t = random_tour(self.n)
			c = tour_length(self.cities, t)
			self.sources.append(FoodSource(tour=t, cost=c, fitness=fitness_from_cost(c)))
		best_initial = min(self.sources, key=lambda s: s.cost)
		self.best = FoodSource(
			tour=best_initial.tour[:],
			cost=best_initial.cost,
			fitness=best_initial.fitness,
			trial=best_initial.trial,
		)
		self.history_best_cost = [self.best.cost]
		self.history_best_tour = [self.best.tour[:]]
		self.improvements = [(0, self.best.cost, self.best.tour[:])]
		self.cycle = 0
		
	def _neighbor(self, tour: List[int]) -> List[int]:
		if self.p.move == "swap":
			return move_swap(tour)
		if self.p.move == "insert":
			return move_insert(tour)
		if self.p.move == "2opt":
			return move_2opt(tour)
		raise ValueError("move debe ser: swap | insert | 2opt")
		
	def _greedy_accept(self, src: FoodSource, cand_tour: List[int]):
		cand_cost = tour_length(self.cities, cand_tour)
		if cand_cost < src.cost:
			src.tour = cand_tour
			src.cost = cand_cost
			src.fitness = fitness_from_cost(cand_cost)
			src.trial = 0
		else:
			src.trial += 1
			
	def _update_best(self):
		cur = min(self.sources, key=lambda s: s.cost)
		if self.best is None or cur.cost < self.best.cost:
			self.best = FoodSource(tour=cur.tour[:], cost=cur.cost, fitness=cur.fitness, trial=cur.trial)
			self.improvements.append((self.cycle, self.best.cost, self.best.tour[:]))
			
		# intensificación opcional sobre el mejor global (didáctico)
		if self.p.intensify_2opt and self.best is not None:
			improved = local_2opt_best_improvement(self.cities, self.best.tour, max_tries=40)
			improved_cost = tour_length(self.cities, improved)
			if improved_cost < self.best.cost:
				self.best.tour = improved
				self.best.cost = improved_cost
				self.best.fitness = fitness_from_cost(improved_cost)
				self.improvements.append((self.cycle, self.best.cost, self.best.tour[:]))
				
	def _probabilities(self) -> List[float]:
		fits = np.array([s.fitness for s in self.sources], dtype=float)
		s = fits.sum()
		if s <= 0:
			return [1.0 / len(self.sources)] * len(self.sources)
		p = (fits / s).tolist()
		return p
		
	def _roulette(self, probs: List[float]) -> int:
		r = random.random()
		acc = 0.0
		for i, p in enumerate(probs):
			acc += p
			if r <= acc:
				return i
		return len(probs) - 1
		
	def step(self):
		self.cycle += 1
		# (A) employed
		for i in range(self.p.SN):
			src = self.sources[i]
			k = random.randrange(self.p.SN)
			while k == i:
				k = random.randrange(self.p.SN)
			# vecino alrededor de src (en discreto, ignoramos k; se podría mezclar tours, pero didáctico)
			cand = self._neighbor(src.tour)
			self._greedy_accept(src, cand)
			
		# (B) onlookers
		probs = self._probabilities()
		for _ in range(self.onlookers):
			i = self._roulette(probs)
			src = self.sources[i]
			cand = self._neighbor(src.tour)
			self._greedy_accept(src, cand)
			
		# (C) scouts
		for i in range(self.p.SN):
			if self.sources[i].trial > self.p.limit:
				t = random_tour(self.n)
				c = tour_length(self.cities, t)
				self.sources[i] = FoodSource(tour=t, cost=c, fitness=fitness_from_cost(c), trial=0)
				
		self._update_best()
		self.history_best_cost.append(self.best.cost if self.best else float("inf"))
		self.history_best_tour.append(self.best.tour[:] if self.best else [])
		
	def run(self):
		for _ in range(self.p.max_cycle):
			self.step()
		return self.best
		
		
# ----------------------------
# Animación
# ----------------------------

def animate_abc_tsp(
    n_cities: int = 30,
    seed: int = 0,
    SN: int = 40,
    max_cycle: int = 400,
    limit: int = 120,
    move: str = "swap",
    intensify_2opt: bool = True,
    output_dir: str = "outputs",
    output_prefix: str = "abc_tsp",
    save_outputs: bool = True,
    show_plot: bool = False,
):
	if save_outputs and not show_plot:
		plt.switch_backend("Agg")

	cities = make_cities(n_cities, seed=seed)
	params = ABCParams(SN=SN, max_cycle=max_cycle, limit=limit, move=move, intensify_2opt=intensify_2opt, seed=seed)
	abc = ABC_TSP(cities, params)

	tracemalloc.start()
	wall_start = time.perf_counter()
	cpu_start = time.process_time()

	for _ in range(max_cycle):
		abc.step()

	wall_end = time.perf_counter()
	cpu_end = time.process_time()
	mem_current, mem_peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	elapsed_wall = wall_end - wall_start
	elapsed_cpu = cpu_end - cpu_start
	peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

	if save_outputs:
		os.makedirs(output_dir, exist_ok=True)
		_save_outputs(
			cities=cities,
			abc=abc,
			params=params,
			output_dir=output_dir,
			output_prefix=output_prefix,
			elapsed_wall=elapsed_wall,
			elapsed_cpu=elapsed_cpu,
			mem_current=mem_current,
			mem_peak=mem_peak,
			peak_rss_kb=peak_rss_kb,
		)

	if show_plot and abc.best is not None:
		fig, ax = plt.subplots()
		ax.set_aspect("equal", adjustable="box")
		ax.set_title("ABC (TSP) - Mejor tour final")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.scatter(cities[:, 0], cities[:, 1])
		tour = abc.best.tour
		xs = [cities[i, 0] for i in tour] + [cities[tour[0], 0]]
		ys = [cities[i, 1] for i in tour] + [cities[tour[0], 1]]
		ax.plot(xs, ys, linewidth=1)
		plt.show()

	return abc.best


def _save_outputs(
	cities: np.ndarray,
	abc: ABC_TSP,
	params: ABCParams,
	output_dir: str,
	output_prefix: str,
	elapsed_wall: float,
	elapsed_cpu: float,
	mem_current: int,
	mem_peak: int,
	peak_rss_kb: int,
):
	best = abc.best
	if best is None:
		return

	# Imagen 1: mejor ruta final
	fig1, ax1 = plt.subplots()
	ax1.set_aspect("equal", adjustable="box")
	ax1.set_title("ABC (TSP) - Mejor ruta final")
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.scatter(cities[:, 0], cities[:, 1])
	xs = [cities[i, 0] for i in best.tour] + [cities[best.tour[0], 0]]
	ys = [cities[i, 1] for i in best.tour] + [cities[best.tour[0], 1]]
	ax1.plot(xs, ys, linewidth=1)
	best_route_img = os.path.join(output_dir, f"{output_prefix}_best_route.png")
	fig1.savefig(best_route_img, dpi=180, bbox_inches="tight")
	plt.close(fig1)

	# Imagen 2: convergencia
	fig2, ax2 = plt.subplots()
	ax2.set_title("ABC (TSP) - Convergencia")
	ax2.set_xlabel("Iteración")
	ax2.set_ylabel("Mejor costo")
	ax2.plot(range(len(abc.history_best_cost)), abc.history_best_cost, linewidth=1)
	conv_img = os.path.join(output_dir, f"{output_prefix}_convergence.png")
	fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
	plt.close(fig2)

	# Reporte de texto plano
	report_index = 1
	while True:
		candidate_report = os.path.join(output_dir, f"{output_prefix}_summary{report_index}.txt")
		if not os.path.exists(candidate_report):
			report_path = candidate_report
			break
		report_index += 1
	with open(report_path, "w", encoding="utf-8") as rep:
		rep.write("ABC para TSP - Resumen de ejecución\n")
		rep.write("=" * 40 + "\n")
		rep.write(f"Ciudades: {cities.shape[0]}\n")
		rep.write(f"SN: {params.SN}\n")
		rep.write(f"Max ciclos: {params.max_cycle}\n")
		rep.write(f"Limit: {params.limit}\n")
		rep.write(f"Movimiento: {params.move}\n")
		rep.write(f"2-opt intensify: {params.intensify_2opt}\n")
		rep.write(f"Seed: {params.seed}\n\n")

		rep.write("Recursos consumidos\n")
		rep.write("-" * 20 + "\n")
		rep.write(f"Tiempo de ejecución (wall): {elapsed_wall:.6f} s\n")
		rep.write(f"Tiempo de CPU: {elapsed_cpu:.6f} s\n")
		rep.write(f"Memoria actual (tracemalloc): {mem_current / (1024 * 1024):.3f} MiB\n")
		rep.write(f"Memoria pico (tracemalloc): {mem_peak / (1024 * 1024):.3f} MiB\n")
		rep.write(f"RSS pico del proceso: {peak_rss_kb} KiB\n\n")

		rep.write("Resultado final\n")
		rep.write("-" * 20 + "\n")
		rep.write(f"Mejor costo final: {best.cost:.6f}\n")
		rep.write(f"Mejor ruta final: {best.tour}\n\n")

		rep.write("Mejores rutas a lo largo de la ejecución\n")
		rep.write("-" * 36 + "\n")
		for cycle, cost, route in abc.improvements:
			rep.write(f"iter={cycle:4d} | cost={cost:.6f} | route={route}\n")

		rep.write("\nArchivos generados\n")
		rep.write("-" * 20 + "\n")
		rep.write(f"{best_route_img}\n")
		rep.write(f"{conv_img}\n")
		rep.write(f"{report_path}\n")


if __name__ == "__main__":
	# Modo sin interfaz gráfica: guarda imágenes y reporte en texto plano.
	
	animate_abc_tsp(
		n_cities=30,
		seed=2,
		SN=50,
		max_cycle=500,
		limit=140,
		move="swap",       # "swap" | "insert" | "2opt"
		intensify_2opt=True,
		output_dir="outputs",
		output_prefix="abc_tsp",
		save_outputs=True,
		show_plot=False,
	)
