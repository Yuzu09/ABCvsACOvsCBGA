"""
Implementación del Algoritmo de Optimización de Colonia de Hormigas (ACO) para el TSP.
"""

import random
import numpy as np


def calcular_distancia_tour(tour, matriz_distancias):
    """
    Calcula la distancia total de un tour (ruta).
    
    Args:
        tour: Lista de índices de ciudades que representan el orden del tour
        matriz_distancias: Matriz de distancias entre ciudades
    
    Returns:
        Distancia total del tour
    """
    distancia = 0
    for i in range(len(tour)):
        ciudad_actual = tour[i]
        ciudad_siguiente = tour[(i + 1) % len(tour)]
        distancia += matriz_distancias[ciudad_actual][ciudad_siguiente]
    return distancia


def construir_solucion_hormiga(num_ciudades, matriz_distancias, matriz_feromonas, alpha, beta):
    """
    Una hormiga construye una solución completa para el TSP.
    
    Args:
        num_ciudades: Número total de ciudades
        matriz_distancias: Matriz de distancias entre ciudades
        matriz_feromonas: Matriz de feromonas entre ciudades
        alpha: Importancia de la feromona
        beta: Importancia de la información heurística (distancia)
    
    Returns:
        Tour completo construido por la hormiga
    """
    # Comenzar desde una ciudad aleatoria
    ciudad_actual = random.randint(0, num_ciudades - 1)
    tour = [ciudad_actual]
    ciudades_no_visitadas = set(range(num_ciudades))
    ciudades_no_visitadas.remove(ciudad_actual)
    
    # Construir tour visitando todas las ciudades
    while ciudades_no_visitadas:
        # Calcular probabilidades para cada ciudad no visitada
        probabilidades = []
        ciudades_disponibles = list(ciudades_no_visitadas)
        
        for ciudad in ciudades_disponibles:
            # Feromona entre ciudad_actual y ciudad
            feromona = matriz_feromonas[ciudad_actual][ciudad] ** alpha
            
            # Información heurística (inverso de la distancia)
            distancia = matriz_distancias[ciudad_actual][ciudad]
            if distancia == 0:
                distancia = 0.0001  # Evitar división por cero
            heuristica = (1.0 / distancia) ** beta
            
            # Probabilidad proporcional a feromona * heurística
            probabilidades.append(feromona * heuristica)
        
        # Normalizar probabilidades
        suma_prob = sum(probabilidades)
        if suma_prob == 0:
            # Si todas las probabilidades son 0, elegir aleatoriamente
            probabilidades = [1.0] * len(probabilidades)
            suma_prob = len(probabilidades)
        
        probabilidades = [p / suma_prob for p in probabilidades]
        
        # Seleccionar siguiente ciudad según probabilidades
        siguiente_ciudad = random.choices(ciudades_disponibles, weights=probabilidades)[0]
        
        tour.append(siguiente_ciudad)
        ciudades_no_visitadas.remove(siguiente_ciudad)
        ciudad_actual = siguiente_ciudad
    
    return tour


def actualizar_feromonas(matriz_feromonas, tours, distancias, rho, q):
    """
    Actualiza la matriz de feromonas según las soluciones encontradas.
    
    Args:
        matriz_feromonas: Matriz actual de feromonas
        tours: Lista de tours construidos por las hormigas
        distancias: Lista de distancias de cada tour
        rho: Tasa de evaporación (0 a 1)
        q: Constante de deposición de feromonas
    """
    # Evaporación de feromonas
    matriz_feromonas *= (1 - rho)
    
    # Depositar feromonas según calidad de soluciones
    for tour, distancia in zip(tours, distancias):
        # Cantidad de feromona a depositar (inversamente proporcional a la distancia)
        deposito = q / distancia
        
        # Depositar feromona en cada arista del tour
        for i in range(len(tour)):
            ciudad_actual = tour[i]
            ciudad_siguiente = tour[(i + 1) % len(tour)]
            matriz_feromonas[ciudad_actual][ciudad_siguiente] += deposito
            matriz_feromonas[ciudad_siguiente][ciudad_actual] += deposito


def algoritmo_colonia_hormigas(matriz_distancias, params):
    """
    Implementa el Algoritmo de Colonia de Hormigas para resolver el TSP.
    
    Args:
        matriz_distancias: Matriz de distancias entre ciudades
        params: Instancia de ParametrosACO con la configuración
    
    Returns:
        Tupla (mejor_ruta, mejor_distancia, historial_fitness)
    """
    num_ciudades = len(matriz_distancias)
    
    # Inicializar matriz de feromonas
    matriz_feromonas = np.ones((num_ciudades, num_ciudades)) * params.feromona_inicial
    
    # Historial para tracking
    historial_fitness = []
    mejor_global = None
    mejor_distancia_global = float('inf')
    
    # Iteraciones del algoritmo
    for iteracion in range(params.iteraciones):
        # Cada hormiga construye una solución
        tours = []
        distancias = []
        
        for _ in range(params.num_hormigas):
            tour = construir_solucion_hormiga(
                num_ciudades, 
                matriz_distancias, 
                matriz_feromonas,
                params.alpha,
                params.beta
            )
            tours.append(tour)
            
            distancia = calcular_distancia_tour(tour, matriz_distancias)
            distancias.append(distancia)
            
            # Actualizar mejor solución global
            if distancia < mejor_distancia_global:
                mejor_distancia_global = distancia
                mejor_global = tour[:]
        
        # Actualizar feromonas
        actualizar_feromonas(
            matriz_feromonas,
            tours,
            distancias,
            params.rho,
            params.q
        )
        
        # Guardar mejor fitness de esta iteración
        historial_fitness.append(mejor_distancia_global)
        
        # Imprimir progreso cada 50 iteraciones
        if (iteracion + 1) % 50 == 0:
            print(f"Iteración {iteracion + 1}/{params.iteraciones} - "
                  f"Mejor distancia: {mejor_distancia_global:.2f}")
    
    return mejor_global, mejor_distancia_global, historial_fitness
