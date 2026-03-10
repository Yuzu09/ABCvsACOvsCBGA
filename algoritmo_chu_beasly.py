import math
import random
import sys
import argparse
from typing import List, Tuple
from typing import Set, Dict, Optional


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
	best = tour
	best_dist = total_distance(best, coords)
	while improved:
		improved = False
		for i in range(1, n - 1):
			for k in range(i + 1, n):
				new_tour = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
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
	# Rotate so smallest node is first; choose direction (forward/reverse) that is lexicographically smaller
	n = len(tour)
	if n == 0:
		return tuple()
	min_idx = tour.index(min(tour))
	rots = [tuple(tour[min_idx:] + tour[:min_idx])]
	# reversed
	r = tour[::-1]
	min_idx_r = r.index(min(r))
	rots.append(tuple(r[min_idx_r:] + r[:min_idx_r]))
	return min(rots)


def tour_edges(tour: List[int]) -> Set[frozenset]:
	return {frozenset((tour[i], tour[(i + 1) % len(tour)])) for i in range(len(tour))}


def diversity_edges(a: List[int], b: List[int]) -> float:
	# D = 1 - |E(a) ∩ E(b)| / n, ranges [0,1]
	n = len(a)
	if n == 0:
		return 0.0
	e_a = tour_edges(a)
	e_b = tour_edges(b)
	common = len(e_a.intersection(e_b))
	return 1.0 - (common / n)


def tournament_selection(pop: List[List[int]], fitness: List[float], k: int = 3) -> List[int]:
	idxs = random.sample(range(len(pop)), k)
	best = min(idxs, key=lambda i: fitness[i])
	return pop[best][:]


def ordered_crossover(p1: List[int], p2: List[int]) -> List[int]:
	# OX crossover
	n = len(p1)
	a, b = sorted(random.sample(range(n), 2))
	child = [-1] * n
	child[a:b+1] = p1[a:b+1]
	pos = (b + 1) % n
	for gene in p2[b+1:] + p2[:b+1]:
		if gene not in child:
			child[pos] = gene
			pos = (pos + 1) % n
	return child


def swap_mutation(tour: List[int], p_mut: float = 0.2) -> List[int]:
	t = tour[:]
	n = len(t)
	for i in range(n):
		if random.random() < p_mut:
			a, b = random.sample(range(n), 2)
			t[a], t[b] = t[b], t[a]
	return t


def initialize_population(coords: List[Tuple[float, float]], pop_size: int, seed: Optional[int] = None, max_tries: int = 50) -> List[List[int]]:
	if seed is not None:
		random.seed(seed)
	n = len(coords)
	population: List[List[int]] = []
	seen: Set[Tuple[int, ...]] = set()
	tries = 0
	while len(population) < pop_size and tries < pop_size * max_tries:
		cand = list(range(n))
		random.shuffle(cand)
		key = canonical_tour(cand)
		if key not in seen:
			seen.add(key)
			population.append(cand)
		tries += 1
	return population


def cbga(coords: List[Tuple[float, float]], pop_size: int = 100, generations: int = 500, min_diversity: float = 0.2,
		p_mut: float = 0.2, p_ls_child: float = 0.1, seed: Optional[int] = None,
		apply_2opt_on_new: bool = True, tournament_k: int = 3) -> Tuple[List[int], float]:
	"""Chu-Beasley style GA (CBGA) with controlled diversity and selective replacement.

	Rules implemented:
	- Population without duplicates (canonical rotation/reverse used for uniqueness)
	- Replacement: child replaces a worse individual only if it improves and respects diversity
	- Diversity metric: edge-difference normalized by n
	- Optional 2-opt intensification on new children
	"""
	if seed is not None:
		random.seed(seed)
	n = len(coords)
	if n == 0:
		return [], 0.0
	# initialize population
	pop = initialize_population(coords, pop_size, seed=seed)
	fitness = [total_distance(t, coords) for t in pop]
	canonical_map: Dict[Tuple[int, ...], int] = {canonical_tour(pop[i]): i for i in range(len(pop))}
	best_idx = min(range(len(pop)), key=lambda i: fitness[i])
	best_tour = pop[best_idx][:]
	best_dist = fitness[best_idx]
	for gen in range(generations):
		# produce one child per generation (replacement-focused GA)
		p1 = tournament_selection(pop, fitness, k=tournament_k)
		p2 = tournament_selection(pop, fitness, k=tournament_k)
		child = ordered_crossover(p1, p2)
		child = swap_mutation(child, p_mut=p_mut)
		# optional local search
		if apply_2opt_on_new and (n <= 200 or random.random() < p_ls_child):
			child = two_opt(child, coords)
		key = canonical_tour(child)
		if key in canonical_map:
			# duplicate, discard
			continue
		child_fit = total_distance(child, coords)
		# S = set of individuals worse than child (higher distance)
		S = [i for i in range(len(pop)) if fitness[i] > child_fit]
		if not S:
			continue
		# choose r in S that is most similar to child (min diversity)
		r = min(S, key=lambda i: diversity_edges(child, pop[i]))
		# check diversity against others (excluding r)
		min_d = min(diversity_edges(child, pop[j]) for j in range(len(pop)) if j != r)
		if min_d + 1e-12 < min_diversity:
			# do not insert if violates min diversity
			continue
		# perform replacement
		del canonical_map[canonical_tour(pop[r])]
		pop[r] = child
		fitness[r] = child_fit
		canonical_map[key] = r
		# update best
		if child_fit < best_dist:
			best_dist = child_fit
			best_tour = child[:]
	# end generations
	return best_tour, best_dist


def chu_beasley(coords: List[Tuple[float, float]], restarts: int = 10, seed: int | None = None) -> Tuple[List[int], float]:
	if seed is not None:
		random.seed(seed)
	n = len(coords)
	if n == 0:
		return [], 0.0
	best_tour = None
	best_dist = float("inf")
	for r in range(restarts):
		start = random.randrange(n)
		tour = nearest_neighbor(coords, start=start)
		tour = two_opt(tour, coords)
		dist = total_distance(tour, coords)
		if dist < best_dist:
			best_dist = dist
			best_tour = tour
	return best_tour, best_dist


def parse_args():
	p = argparse.ArgumentParser(description="Algoritmo Chu-Beasley (heurística TSP) y CBGA (GA controlado)")
	p.add_argument("instance", nargs="?", default="berlin52.tsp", help="Archivo .tsp (EUC_2D)")
	# heuristic options
	p.add_argument("--restarts", type=int, default=20, help="Número de reinicios aleatorios (heurística)")
	# GA options
	p.add_argument("--ga", action="store_true", help="Ejecutar CBGA (genetic algorithm controlado)")
	p.add_argument("--pop", type=int, default=100, help="Tamaño de población para CBGA")
	p.add_argument("--generations", type=int, default=500, help="Generaciones para CBGA")
	p.add_argument("--min-diversity", type=float, default=0.2, help="Umbral mínimo de diversidad (0-1)")
	p.add_argument("--p-mut", type=float, default=0.2, help="Probabilidad de mutación por gen (swap)")
	p.add_argument("--p-ls-child", type=float, default=0.1, help="Probabilidad de aplicar 2-opt a hijos nuevos (si n>200)")
	p.add_argument("--apply-2opt", action="store_true", help="Aplicar 2-opt a hijos nuevos cuando proceda")
	p.add_argument("--tournament-k", type=int, default=3, help="Tamaño del torneo para selección")
	p.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
	return p.parse_args()


def main():
	args = parse_args()
	coords = parse_tsp(args.instance)
	if not coords:
		print("No se encontraron coordenadas en la instancia.")
		sys.exit(1)
	if args.ga:
		print("Ejecutando CBGA (GA controlado) ...")
		tour, dist = cbga(coords, pop_size=args.pop, generations=args.generations,
			min_diversity=args.min_diversity, p_mut=args.p_mut, p_ls_child=args.p_ls_child,
			seed=args.seed, apply_2opt_on_new=args.apply_2opt, tournament_k=args.tournament_k)
	else:
		tour, dist = chu_beasley(coords, restarts=args.restarts, seed=args.seed)

	print(f"Instancia: {args.instance}")
	print(f"Nodos: {len(coords)}")
	print(f"Distancia mejor tour: {dist:.4f}")
	print("Tour (0-based indices):")
	print(tour)


if __name__ == "__main__":
	main()

