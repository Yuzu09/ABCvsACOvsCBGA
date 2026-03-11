"""
Implementación ACO para TSP con salida headless (imágenes + resumen de texto).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import argparse
import os
import random
import time
import tracemalloc
import resource

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ACOConfig:
    num_hormigas: int = 5
    alpha: float = 1.0
    beta: float = 5.0
    rho: float = 0.3
    q: float = 100.0
    iteraciones: int = 300
    feromona_inicial: float = 1.0
    seed: int = 11


class ACOTSPSolver:
    def __init__(self, config: ACOConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)

    @staticmethod
    def calcular_distancia_tour(tour: List[int], matriz_distancias: np.ndarray) -> float:
        distancia = 0.0
        for i in range(len(tour)):
            ciudad_actual = tour[i]
            ciudad_siguiente = tour[(i + 1) % len(tour)]
            distancia += matriz_distancias[ciudad_actual][ciudad_siguiente]
        return float(distancia)

    def _construir_solucion_hormiga(
        self,
        num_ciudades: int,
        matriz_distancias: np.ndarray,
        matriz_feromonas: np.ndarray,
    ) -> List[int]:
        ciudad_actual = self.rng.randint(0, num_ciudades - 1)
        tour = [ciudad_actual]
        ciudades_no_visitadas = set(range(num_ciudades))
        ciudades_no_visitadas.remove(ciudad_actual)

        while ciudades_no_visitadas:
            probabilidades = []
            ciudades_disponibles = list(ciudades_no_visitadas)

            for ciudad in ciudades_disponibles:
                feromona = matriz_feromonas[ciudad_actual][ciudad] ** self.config.alpha
                distancia = matriz_distancias[ciudad_actual][ciudad]
                if distancia == 0:
                    distancia = 1e-4
                heuristica = (1.0 / distancia) ** self.config.beta
                probabilidades.append(feromona * heuristica)

            suma_prob = sum(probabilidades)
            if suma_prob == 0:
                probabilidades = [1.0] * len(probabilidades)
                suma_prob = float(len(probabilidades))

            probabilidades = [p / suma_prob for p in probabilidades]
            siguiente_ciudad = self.rng.choices(ciudades_disponibles, weights=probabilidades, k=1)[0]

            tour.append(siguiente_ciudad)
            ciudades_no_visitadas.remove(siguiente_ciudad)
            ciudad_actual = siguiente_ciudad

        return tour

    def _actualizar_feromonas(
        self,
        matriz_feromonas: np.ndarray,
        tours: List[List[int]],
        distancias: List[float],
    ) -> None:
        matriz_feromonas *= (1.0 - self.config.rho)

        for tour, distancia in zip(tours, distancias):
            if distancia <= 0:
                continue
            deposito = self.config.q / distancia
            for i in range(len(tour)):
                ciudad_actual = tour[i]
                ciudad_siguiente = tour[(i + 1) % len(tour)]
                matriz_feromonas[ciudad_actual][ciudad_siguiente] += deposito
                matriz_feromonas[ciudad_siguiente][ciudad_actual] += deposito

    def solve(self, matriz_distancias: np.ndarray) -> Dict[str, Any]:
        num_ciudades = len(matriz_distancias)
        matriz_feromonas = np.ones((num_ciudades, num_ciudades), dtype=float) * self.config.feromona_inicial

        historial_fitness: List[float] = []
        mejoras: List[Tuple[int, float, List[int]]] = []
        mejor_global: List[int] | None = None
        mejor_distancia_global = float("inf")

        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        for iteracion in range(self.config.iteraciones):
            tours: List[List[int]] = []
            distancias: List[float] = []

            for _ in range(self.config.num_hormigas):
                tour = self._construir_solucion_hormiga(num_ciudades, matriz_distancias, matriz_feromonas)
                distancia = self.calcular_distancia_tour(tour, matriz_distancias)
                tours.append(tour)
                distancias.append(distancia)

                if distancia < mejor_distancia_global:
                    mejor_distancia_global = distancia
                    mejor_global = tour[:]
                    mejoras.append((iteracion + 1, mejor_distancia_global, mejor_global[:]))

            self._actualizar_feromonas(matriz_feromonas, tours, distancias)
            historial_fitness.append(mejor_distancia_global)

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "mejor_ruta": mejor_global,
            "mejor_distancia": mejor_distancia_global,
            "historial_fitness": historial_fitness,
            "mejoras": mejoras,
            "tiempo_wall": wall_end - wall_start,
            "tiempo_cpu": cpu_end - cpu_start,
            "memoria_actual": mem_current,
            "memoria_pico": mem_peak,
            "rss_pico_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "config": self.config,
        }


def _infer_coordinates_from_distance_matrix(matriz_distancias: np.ndarray, seed: int = 0) -> np.ndarray:
    n = len(matriz_distancias)
    rng = np.random.default_rng(seed)
    return rng.random((n, 2), dtype=float)


def _save_outputs(
    matriz_distancias: np.ndarray,
    resultado: Dict[str, Any],
    output_dir: str,
    output_prefix: str,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    mejor_ruta = resultado["mejor_ruta"]
    historial = resultado["historial_fitness"]

    # Imagen 1: mejor ruta (layout sintético para visualizar en TSP genérico)
    coords = _infer_coordinates_from_distance_matrix(matriz_distancias, seed=seed)
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("ACO (TSP) - Mejor ruta final")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(coords[:, 0], coords[:, 1])

    if mejor_ruta:
        xs = [coords[i, 0] for i in mejor_ruta] + [coords[mejor_ruta[0], 0]]
        ys = [coords[i, 1] for i in mejor_ruta] + [coords[mejor_ruta[0], 1]]
        ax1.plot(xs, ys, linewidth=1)

    best_route_img = os.path.join(output_dir, f"{output_prefix}_best_route.png")
    fig1.savefig(best_route_img, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    # Imagen 2: convergencia
    fig2, ax2 = plt.subplots()
    ax2.set_title("ACO (TSP) - Convergencia")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Mejor distancia")
    ax2.plot(range(1, len(historial) + 1), historial, linewidth=1)
    conv_img = os.path.join(output_dir, f"{output_prefix}_convergence.png")
    fig2.savefig(conv_img, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Reporte incremental summaryN
    report_index = 1
    while True:
        candidate_report = os.path.join(output_dir, f"{output_prefix}_summary{report_index}.txt")
        if not os.path.exists(candidate_report):
            report_path = candidate_report
            break
        report_index += 1

    config: ACOConfig = resultado["config"]
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("ACO para TSP - Resumen de ejecución\n")
        rep.write("=" * 40 + "\n")
        rep.write(f"Ciudades: {len(matriz_distancias)}\n")
        rep.write(f"Hormigas: {config.num_hormigas}\n")
        rep.write(f"Alpha: {config.alpha}\n")
        rep.write(f"Beta: {config.beta}\n")
        rep.write(f"Rho: {config.rho}\n")
        rep.write(f"Q: {config.q}\n")
        rep.write(f"Iteraciones: {config.iteraciones}\n")
        rep.write(f"Feromona inicial: {config.feromona_inicial}\n")
        rep.write(f"Seed: {config.seed}\n\n")

        rep.write("Recursos consumidos\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"Tiempo de ejecución (wall): {resultado['tiempo_wall']:.6f} s\n")
        rep.write(f"Tiempo de CPU: {resultado['tiempo_cpu']:.6f} s\n")
        rep.write(f"Memoria actual (tracemalloc): {resultado['memoria_actual'] / (1024 * 1024):.3f} MiB\n")
        rep.write(f"Memoria pico (tracemalloc): {resultado['memoria_pico'] / (1024 * 1024):.3f} MiB\n")
        rep.write(f"RSS pico del proceso: {resultado['rss_pico_kb']} KiB\n\n")

        rep.write("Resultado final\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"Mejor distancia final: {resultado['mejor_distancia']:.6f}\n")
        rep.write(f"Mejor ruta final: {resultado['mejor_ruta']}\n\n")

        rep.write("Mejores rutas a lo largo de la ejecución\n")
        rep.write("-" * 36 + "\n")
        for iteracion, distancia, ruta in resultado["mejoras"]:
            rep.write(f"iter={iteracion:4d} | dist={distancia:.6f} | route={ruta}\n")

        rep.write("\nArchivos generados\n")
        rep.write("-" * 20 + "\n")
        rep.write(f"{best_route_img}\n")
        rep.write(f"{conv_img}\n")
        rep.write(f"{report_path}\n")


def ejecutar_aco_tsp(
    matriz_distancias: np.ndarray,
    config: ACOConfig | None = None,
    output_dir: str = "outputs",
    output_prefix: str = "aco_tsp",
    save_outputs: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    if config is None:
        config = ACOConfig()

    if save_outputs and not show_plot:
        plt.switch_backend("Agg")

    solver = ACOTSPSolver(config)
    resultado = solver.solve(matriz_distancias)

    if save_outputs:
        _save_outputs(
            matriz_distancias=matriz_distancias,
            resultado=resultado,
            output_dir=output_dir,
            output_prefix=output_prefix,
            seed=config.seed,
        )

    if show_plot and resultado["mejor_ruta"]:
        coords = _infer_coordinates_from_distance_matrix(matriz_distancias, seed=config.seed)
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("ACO (TSP) - Mejor ruta final")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(coords[:, 0], coords[:, 1])
        ruta = resultado["mejor_ruta"]
        xs = [coords[i, 0] for i in ruta] + [coords[ruta[0], 0]]
        ys = [coords[i, 1] for i in ruta] + [coords[ruta[0], 1]]
        ax.plot(xs, ys, linewidth=1)
        plt.show()

    return resultado


def algoritmo_colonia_hormigas(matriz_distancias, params):
    """
    Envoltura de compatibilidad con la interfaz anterior.
    Retorna: (mejor_ruta, mejor_distancia, historial_fitness)
    """
    config = ACOConfig(
        num_hormigas=getattr(params, "num_hormigas", 20),
        alpha=getattr(params, "alpha", 1.0),
        beta=getattr(params, "beta", 3.0),
        rho=getattr(params, "rho", 0.3),
        q=getattr(params, "q", 100.0),
        iteraciones=getattr(params, "iteraciones", 120),
        feromona_inicial=getattr(params, "feromona_inicial", 1.0),
        seed=getattr(params, "seed", 11),
    )

    resultado = ejecutar_aco_tsp(
        matriz_distancias=np.array(matriz_distancias, dtype=float),
        config=config,
        output_dir="outputs",
        output_prefix="aco_tsp",
        save_outputs=True,
        show_plot=False,
    )

    return resultado["mejor_ruta"], resultado["mejor_distancia"], resultado["historial_fitness"]


def _build_random_distance_matrix(n_ciudades: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_ciudades, 2), dtype=float)
    matriz = np.zeros((n_ciudades, n_ciudades), dtype=float)
    for i in range(n_ciudades):
        for j in range(n_ciudades):
            matriz[i, j] = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
    return matriz


def parse_args():
    parser = argparse.ArgumentParser(description="ACO para TSP con salidas headless")
    parser.add_argument("--n-ciudades", type=int, default=30, help="Cantidad de ciudades para instancia aleatoria")
    parser.add_argument("--num-hormigas", type=int, default=5, help="Número de hormigas")
    parser.add_argument("--alpha", type=float, default=1.0, help="Peso de feromona")
    parser.add_argument("--beta", type=float, default=5.0, help="Peso heurístico")
    parser.add_argument("--rho", type=float, default=0.3, help="Tasa de evaporación")
    parser.add_argument("--q", type=float, default=100.0, help="Constante de depósito")
    parser.add_argument("--iteraciones", type=int, default=300, help="Número de iteraciones")
    parser.add_argument("--feromona-inicial", type=float, default=1.0, help="Valor inicial de feromona")
    parser.add_argument("--seed", type=int, default=11, help="Semilla aleatoria")
    parser.add_argument("--output-dir", default="outputs", help="Carpeta de salida")
    parser.add_argument("--output-prefix", default="aco_tsp", help="Prefijo de archivos")
    parser.add_argument("--show-plot", action="store_true", help="Mostrar figura en pantalla")
    return parser.parse_args()


def main():
    args = parse_args()

    matriz_distancias = _build_random_distance_matrix(args.n_ciudades, args.seed)
    config = ACOConfig(
        num_hormigas=args.num_hormigas,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q=args.q,
        iteraciones=args.iteraciones,
        feromona_inicial=args.feromona_inicial,
        seed=args.seed,
    )

    resultado = ejecutar_aco_tsp(
        matriz_distancias=matriz_distancias,
        config=config,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        save_outputs=True,
        show_plot=args.show_plot,
    )

    print("Ejecución ACO completada")
    print(f"Mejor distancia: {resultado['mejor_distancia']:.6f}")
    print(f"Mejor ruta: {resultado['mejor_ruta']}")
    print(f"Convergencia: {os.path.join(args.output_dir, args.output_prefix + '_convergence.png')}")
    print(f"Mejor ruta (imagen): {os.path.join(args.output_dir, args.output_prefix + '_best_route.png')}")
    print("Resumen: archivo incremental *_summaryN.txt en outputs")


if __name__ == "__main__":
    main()
