# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       DE/ed.py
# Descripción:   Implementación mejorada de Evolución Diferencial según
#                [Carmona&Galán-2020].
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random
from datetime import datetime
from base_algorithm import BaseEvolutionaryAlgorithm

@dataclass
class Individual:
    """Representa un individuo en ED."""
    x: np.ndarray
    fitness: float = float('inf')

class ImprovedDE(BaseEvolutionaryAlgorithm):
    """
    Implementación mejorada de Evolución Diferencial según [Carmona&Galán-2020].

    Implementa el algoritmo canónico (sección 5.2) con las variantes:
    - DE/rand/1/bin: variante clásica con vector base aleatorio (sección 5.4)
    - DE/best/1/bin: variante con mejor vector como base (sección 5.9.1)

    Atributos:
        n_dim (int): Número de variables a optimizar
        pop_size (int): Tamaño de población (TP)
        variant (str): Variante de ED ('rand/1/bin' o 'best/1/bin')
        F (float): Peso del diferencial en [0,1]
        CR (float): Probabilidad de cruce en [0,1]
        x_min (float): Límite inferior del espacio de búsqueda
        x_max (float): Límite superior del espacio de búsqueda
        Pcr (float): Probabilidad de la estrategia either-or
    """
    def __init__(self, n_dim: int = 10, pop_size: int = 30,
                 variant: str = 'rand/1/bin', f: float = 0.8,
                 cr: float = 0.9, x_min: float = -100,
                 x_max: float = 100, seed: int = None,
                 diversity_threshold: float = 1e-6,
                 local_search_prob: float = 0.1,
                 pcr: float = 0.4):

        if not 0 <= f <= 1 or not 0 <= cr <= 1:
            raise ValueError("F y CR deben estar en [0,1]")

        self.n_dim = n_dim
        self._population_size = pop_size
        if variant not in ['rand/1/bin', 'best/1/bin']:
            raise ValueError("Variante debe ser 'rand/1/bin' o 'best/1/bin'")
        self.variant = variant
        self.F = f
        self.initial_F = f
        self.CR = cr
        self.initial_CR = cr
        self.x_min = x_min
        self.x_max = x_max
        self._seed = seed
        self.diversity_threshold = diversity_threshold
        self.local_search_prob = local_search_prob
        self.Pcr = pcr

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    @property
    def evaluations_per_generation(self) -> int:
        return self.population_size

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def population_size(self) -> int:
        return self._population_size

    def init_population(self, fitness_func: callable) -> List[Individual]:
        """
        Inicialización de la población según sección 5.5.
        Genera TP individuos distribuidos uniformemente en el espacio de búsqueda.
        """
        population = []
        segments = np.linspace(self.x_min, self.x_max, self.population_size + 1)
        for i in range(self.population_size):
            x = np.array([np.random.uniform(segments[i], segments[i+1])
                         for _ in range(self.n_dim)])
            ind = Individual(x=x)
            ind.fitness = float(fitness_func(x))
            population.append(ind)
        return sorted(population, key=lambda individual: individual.fitness)

    def create_mutant_vector(self, population: List[Individual],
                           i: int, current_best: Individual) -> np.ndarray:
        """
        Implementa la mutación diferencial según secciones 5.4 y 5.9.1.
        Para DE/rand/1/bin: v = xr + F*(xp - xq)
        Para DE/best/1/bin: v = x_best + F*(xp - xq)
        """
        if self.variant == 'rand/1/bin':
            r1, r2, r3 = random.sample([j for j in range(self.population_size) if j != i], 3)
            return population[r1].x + self.F * (population[r2].x - population[r3].x)
        else:  # 'best/1/bin'
            r1, r2 = random.sample(range(self.population_size), 2)
            return current_best.x + self.F * (population[r1].x - population[r2].x)

    def local_search(self, x: np.ndarray, fitness_func: callable) -> Tuple[np.ndarray, float]:
        """
        Búsqueda local para intensificar la exploración cerca del óptimo.
        Implementación basada en las recomendaciones de la sección 5.10.
        """
        best_x, best_fitness = x.copy(), float(fitness_func(x))

        if best_fitness < 500:
            radii = [0.005, 0.001, 0.0001]
            n_attempts = 15
        else:
            radii = [0.1, 0.01, 0.001]
            n_attempts = 5

        for radius in radii:
            scale = radius * (self.x_max - self.x_min)
            for _ in range(n_attempts):
                x_new = best_x + np.random.normal(0, scale, self.n_dim)
                x_new = np.clip(x_new, self.x_min, self.x_max)
                fitness_new = float(fitness_func(x_new))
                if fitness_new < best_fitness:
                    best_x, best_fitness = x_new, fitness_new

        return best_x, best_fitness

    def check_diversity(self, population: List[Individual]) -> bool:
        """
        Calcula la diversidad de la población como la distancia media entre individuos.
        Ayuda a detectar convergencia prematura según sección 5.10.
        """
        distances = []
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                dist = float(np.linalg.norm(population[i].x - population[j].x))
                distances.append(dist)
        return float(np.mean(distances)) > self.diversity_threshold

    def adapt_parameters(self, success_rate: float, gen: int, max_gen: int,
                        current_best_fitness: float):
        """
        Adaptación de parámetros F y CR según recomendaciones de sección 5.10.3.
        Incluye la estrategia either-or de la sección 5.9.3.
        """
        if current_best_fitness < 500:
            self.F = max(0.05, min(0.3, self.F))
            self.CR = 0.98
            self.Pcr = min(0.6, self.Pcr * 1.1)
        else:
            if gen < max_gen * 0.3:
                self.F = max(0.6, min(0.9, self.F))
                self.CR = 0.9
            elif gen < max_gen * 0.7:
                self.F = max(0.4, min(0.7, self.F))
                self.CR = 0.8
            else:
                self.F = max(0.2, min(0.5, self.F))
                self.CR = 0.7
            self.Pcr = max(0.3, self.Pcr * 0.95)

        if success_rate < 0.2:
            self.F *= 0.97
        elif success_rate > 0.8:
            self.F = min(1.0, self.F * 1.03)

    def reinitialize_population(self, population: List[Individual],
                              fitness_func: callable) -> List[Individual]:
        """
        Reinicialización de la población cuando se detecta convergencia prematura.
        Sigue las recomendaciones de inicialización de la sección 5.10.1.
        """
        if not self.check_diversity(population):
            best_fitness = float(population[0].fitness)
            if best_fitness < 500:
                n_keep = max(5, self.population_size // 3)
                kept_individuals = population[:n_keep]
                scale = 0.01
            else:
                n_keep = max(3, self.population_size // 4)
                kept_individuals = population[:n_keep]
                scale = 0.05

            new_individuals = []
            for _ in range(self.population_size - n_keep):
                if random.random() < 0.5:
                    x = np.array([random.uniform(self.x_min, self.x_max)
                                 for _ in range(self.n_dim)])
                else:
                    parents = random.sample(kept_individuals, 2)
                    alpha = random.random()
                    x = alpha * parents[0].x + (1-alpha) * parents[1].x + \
                        np.random.normal(0, scale, self.n_dim)
                x = np.clip(x, self.x_min, self.x_max)
                new_individuals.append(Individual(x=x, fitness=float(fitness_func(x))))

            return sorted(kept_individuals + new_individuals,
                         key=lambda ind: ind.fitness)
        return population

    def evolve(self, fitness_func: callable, max_gen: int,
               recombine_type: str = 'intermediate',
               selection_type: str = 'comma'
               ) -> Tuple[List[float], np.ndarray, dict]:
        """
        Implementa el algoritmo ED con estrategia either-or según sección 5.9.3.
        En cada generación:
        1. Para cada individuo xi:
           - Con probabilidad Pcr usa recombinación avanzada
           - En otro caso usa mutación diferencial clásica y cruce binomial
        2. Aplica selección de supervivientes
        3. Actualiza parámetros F y CR
        """
        population = self.init_population(fitness_func)
        best_fitness_history = []
        best_x = population[0].x.copy()
        history = {'generations': [], 'populations': [], 'timestamps': []}
        stagnation_counter = 0
        prev_best = float('inf')

        history['populations'].append([{'x': ind.x.copy(), 'fitness': float(ind.fitness)}
                                     for ind in population])
        history['timestamps'].append(datetime.now())

        for gen in range(max_gen):
            success_count = 0
            new_population = []
            current_best = population[0]
            _K = 0.5 * (self.population_size + 1)

            for i in range(self.population_size):
                if random.random() < self.Pcr:
                    # Estrategia de recombinación
                    r1, r2 = random.sample([j for j in range(self.population_size) if j != i], 2)
                    u = current_best.x + _K * (population[r1].x + population[r2].x - 2*current_best.x)
                else:
                    # Estrategia de mutación diferencial clásica
                    v = self.create_mutant_vector(population, i, current_best)
                    u = np.zeros(self.n_dim)
                    j_rand = random.randint(0, self.n_dim - 1)

                    for j in range(self.n_dim):
                        if random.random() < self.CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = population[i].x[j]

                u = np.clip(u, self.x_min, self.x_max)
                u_fitness = float(fitness_func(u))

                if u_fitness < population[i].fitness:
                    new_population.append(Individual(x=u, fitness=u_fitness))
                    success_count += 1
                else:
                    new_population.append(Individual(x=population[i].x.copy(),
                                                  fitness=float(population[i].fitness)))

            local_search_prob = self.local_search_prob * (1.0 + max(0.0, (500.0 - float(current_best.fitness)))/500.0)
            if random.random() < local_search_prob:
                best_x_new, best_fitness_new = self.local_search(current_best.x, fitness_func)
                if best_fitness_new < current_best.fitness:
                    new_population[0] = Individual(x=best_x_new, fitness=float(best_fitness_new))

            new_population.sort(key=lambda ind: ind.fitness)

            if abs(float(new_population[0].fitness) - float(prev_best)) < 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best = float(new_population[0].fitness)

            if int(stagnation_counter) > 50:
                population = self.reinitialize_population(new_population, fitness_func)
                stagnation_counter = 0
            else:
                population = new_population

            self.adapt_parameters(success_count / self.population_size, gen, max_gen,
                                float(population[0].fitness))

            if (not best_fitness_history or
                float(population[0].fitness) < float(best_fitness_history[-1])):
                best_x = population[0].x.copy()
            best_fitness_history.append(float(population[0].fitness))

            history['generations'].append(gen)
            history['populations'].append([{'x': ind.x.copy(), 'fitness': float(ind.fitness)}
                                         for ind in population])
            history['timestamps'].append(datetime.now())

            if gen % 100 == 0:
                print(f"Generación {gen}: Mejor fitness = {float(population[0].fitness):.6e}")

        return best_fitness_history, best_x, history