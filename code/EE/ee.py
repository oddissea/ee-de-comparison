# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       EE/ee.py
# Descripción:   Implementación de Estrategias Evolutivas según
#                [Carmona&Galán-2020].
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from base_algorithm import BaseEvolutionaryAlgorithm
import random
from datetime import datetime


@dataclass
class Individual:
    """Representa un individuo en la EE."""
    x: np.ndarray
    sigma: np.ndarray
    fitness: float = float('inf')


class ImprovedEE(BaseEvolutionaryAlgorithm):
    """
    Implementación de una Estrategia Evolutiva según [Carmona&Galán-2020].

    Esta implementación sigue el algoritmo canónico descrito en la sección 3.2 del libro,
    permitiendo tanto, mutación no correlacionada de 1-paso, como de n-pasos (secciones 3.7.2 y 3.7.3).

    Atributos:
        n_dim (int): Número de variables a optimizar
        n_sigma (int): Número de parámetros sigma (1 o n_dim)
        mu (int): Tamaño de la población de padres (μ)
        lambda_ (int): Tamaño de la población de descendientes (λ)
        epsilon_0 (float): Valor mínimo permitido para sigma
        x_min (float): Límite inferior del espacio de búsqueda
        x_max (float): Límite superior del espacio de búsqueda
        rho (int): Número de padres para recombinación (ρ)
    """
    @property
    def evaluations_per_generation(self) -> int:
        return self.lambda_

    @property
    def seed(self) -> int:
        """Semilla para reproducibilidad"""
        return self._seed

    @property
    def population_size(self) -> int:
        """Tamaño de la población para EE: mu"""
        return self.mu

    def __init__(self, n_dim: int = 10, n_sigma: int = 1, mu: int = 30,
                 lambda_: int = 200, epsilon_0: float = 1e-5,
                 x_min: float = -100, x_max: float = 100, rho: int = 2,
                 seed: int = None):
        self.n_dim = n_dim
        self.n_sigma = n_sigma
        self.mu = mu
        self.lambda_ = lambda_
        self.epsilon_0 = epsilon_0
        self.x_min = x_min
        self.x_max = x_max
        self.rho = rho
        self._seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if self.lambda_ / self.mu < 7:
            print("Advertencia: Se recomienda λ/μ = 7")
        if self.mu < 15:
            print("Advertencia: Se recomienda μ >> 1 (por ejemplo, μ = 15)")
        if self.n_sigma not in [1, self.n_dim]:
            raise ValueError("n_sigma debe ser 1 o n_dim")

        if self.n_sigma == 1:
            self.tau = 1.0 / np.sqrt(self.n_dim)
            self.tau_prime = None
        else:
            self.tau_prime = 1.0 / np.sqrt(2.0 * self.n_dim)
            self.tau = 1.0 / np.sqrt(2.0 * np.sqrt(self.n_dim))


    @staticmethod
    def sphere_function(x: np.ndarray) -> float:
        return float(np.sum((x - 10.0) ** 2))

    @staticmethod
    def schwefel_function(x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n + np.sum(-x * np.sin(np.sqrt(np.abs(x))))

    def init_population(self, fitness_func: callable) -> List[Individual]:
        """
        Inicializa la población según la sección 3.4 del libro.
        Se generan μ individuos con variables aleatorias uniformemente distribuidas
        en el espacio de búsqueda y valores sigma inicializados a 1.

        Args:
            fitness_func: Función de evaluación
        Returns:
            Lista de μ individuos inicializados
        """
        population = []
        for _ in range(self.mu):
            x = np.random.uniform(self.x_min, self.x_max, self.n_dim)
            sigma = np.ones(self.n_sigma)
            ind = Individual(x=x, sigma=sigma)
            ind.fitness = fitness_func(x)
            population.append(ind)
        return population

    @staticmethod
    def select_parents(current_pop: List[Individual], num_parents: int) -> List[Individual]:
        """
        Implementa la selección de padres según sección 3.5.
        La selección es aleatoria uniforme sin sesgo por el valor de adaptación.

        Args:
            current_pop: Población actual
            num_parents: Número de padres a seleccionar (ρ)
        Returns:
            Lista de ρ padres seleccionados
        """
        selected_indices = random.sample(range(len(current_pop)), num_parents)
        return [current_pop[idx] for idx in selected_indices]

    def recombine_discrete_global(self, selected_parents: List[Individual]) -> Individual:
        """
        Implementa la recombinación discreta global según sección 3.6.
        Para cada gen del hijo, se selecciona aleatoriamente el valor
        del gen correspondiente de uno de los padres.

        Args:
            selected_parents: Lista de padres seleccionados
        Returns:
            Nuevo individuo generado por recombinación discreta
        """
        child_x = np.zeros(self.n_dim)
        child_sigma = np.zeros(self.n_sigma)

        for i in range(self.n_dim):
            parent_idx = np.random.randint(0, len(selected_parents))
            child_x[i] = selected_parents[parent_idx].x[i]

        for i in range(self.n_sigma):
            parent_idx = np.random.randint(0, len(selected_parents))
            child_sigma[i] = selected_parents[parent_idx].sigma[i]

        return Individual(x=child_x, sigma=child_sigma)

    @staticmethod
    def recombine_intermediate_global(selected_parents: List[Individual]) -> Individual:
        """
        Implementa la recombinación intermedia global según sección 3.6.
        El valor de cada gen del hijo es el promedio de los valores
        del gen correspondiente de todos los padres.

        Args:
            selected_parents: Lista de padres seleccionados
        Returns:
            Nuevo individuo generado por recombinación intermedia
        """
        child_x = np.mean([p.x for p in selected_parents], axis=0)
        child_sigma = np.mean([p.sigma for p in selected_parents], axis=0)
        return Individual(x=child_x, sigma=child_sigma)

    def recombine(self, selected_parents: List[Individual], recombine_type: str = 'intermediate') -> Individual:
        """
        Selecciona el tipo de recombinación a aplicar según la sección 3.6.
        Permite elegir entre recombinación discreta o intermedia global.

        Args:
            selected_parents: Lista de padres seleccionados
            recombine_type: Tipo de recombinación ('discrete' o 'intermediate')
        Returns:
            Nuevo individuo generado por recombinación
        Raises:
            ValueError: Si el tipo de recombinación no es válido
        """
        if recombine_type == 'discrete':
            return self.recombine_discrete_global(selected_parents)
        elif recombine_type == 'intermediate':
            return self.recombine_intermediate_global(selected_parents)
        else:
            raise ValueError(f"Tipo de recombinación no válido: {recombine_type}")

    def mutate_one_step(self, ind: Individual) -> Individual:
        """
        Implementa la mutación no correlacionada de 1-paso según sección 3.7.2.
        Utiliza un único parámetro sigma que se adapta mediante una distribución
        log-normal. Las variables se mutan añadiendo perturbaciones normales.

        Args:
            ind: Individuo a mutar
        Returns:
            Individuo mutado
        """
        if len(ind.sigma) != 1:
            raise ValueError("Para mutación de 1-paso, el individuo debe tener un único sigma")

        n0 = np.random.normal(0, 1)
        sigma_prime = ind.sigma[0] * np.exp(self.tau * n0)

        if sigma_prime < self.epsilon_0:
            sigma_prime = self.epsilon_0

        x_prime = np.array([ind.x[i] + sigma_prime * np.random.normal(0, 1)
                            for i in range(self.n_dim)])
        x_prime = np.clip(x_prime, self.x_min, self.x_max)

        return Individual(x=x_prime, sigma=np.array([sigma_prime]))

    def mutate_n_step(self, ind: Individual) -> Individual:
        """
        Implementa la mutación no correlacionada de n-pasos según sección 3.7.3.
        Utiliza n_dim parámetros sigma que se adaptan mediante distribuciones
        log-normales. Las variables se mutan añadiendo perturbaciones normales.

        Args:
            ind: Individuo a mutar
        Returns:
            Individuo mutado
        """

        if len(ind.sigma) != self.n_dim:
            raise ValueError("Para mutación de n-pasos, el número de sigmas debe ser "
                             "igual al número de dimensiones")

        n0 = np.random.normal(0, 1)
        sigma_prime = np.array([
            sig * np.exp(self.tau_prime * n0 + self.tau * np.random.normal(0, 1))
            for sig in ind.sigma
        ])

        sigma_prime[sigma_prime < self.epsilon_0] = self.epsilon_0

        x_prime = np.array([ind.x[i] + sigma_prime[i] * np.random.normal(0, 1)
                            for i in range(self.n_dim)])
        x_prime = np.clip(x_prime, self.x_min, self.x_max)

        return Individual(x=x_prime, sigma=sigma_prime)

    def select_survivors_comma(self, offspring: List[Individual]) -> List[Individual]:
        """
        Implementa la selección (μ,λ) según sección 3.8.
        Selecciona los μ mejores individuos solo de entre los λ descendientes.
        Aplica una fuerte presión selectiva al ser λ >> μ.

        Args:
            offspring: Lista de λ descendientes
        Returns:
            Los μ mejores descendientes
        """
        sorted_offspring = sorted(offspring, key=lambda x: x.fitness)
        return sorted_offspring[:self.mu]

    def select_survivors_plus(self, parents: List[Individual],
                              offspring: List[Individual]) -> List[Individual]:
        """
        Implementa la selección (μ+λ) según sección 3.8.
        Selecciona los μ mejores individuos de entre padres y descendientes.

        Args:
            parents: Lista de padres
            offspring: Lista de descendientes
        Returns:
            Los μ mejores individuos
        """

        all_individuals = parents + offspring
        sorted_individuals = sorted(all_individuals, key=lambda x: x.fitness)
        return sorted_individuals[:self.mu]

    def select_survivors(self, parents: List[Individual],
                         offspring: List[Individual],
                         selection_type: str = 'comma') -> List[Individual]:
        """
        Selecciona el tipo de selección de supervivientes según sección 3.8.
        Permite elegir entre selección (μ,λ) o (μ+λ).

        Args:
            parents: Lista de μ padres
            offspring: Lista de λ descendientes
            selection_type: Tipo de selección ('comma' o 'plus')
        Returns:
            Los μ mejores individuos según el criterio elegido
        Raises:
            ValueError: Si el tipo de selección no es válido
        """
        if selection_type == 'comma':
            return self.select_survivors_comma(offspring)
        elif selection_type == 'plus':
            return self.select_survivors_plus(parents, offspring)
        else:
            raise ValueError(f"Tipo de selección no válido: {selection_type}")


    def evolve(self, fitness_func: callable, max_gen: int,
               recombine_type: str = 'intermediate',
               selection_type: str = 'comma') -> Tuple[List[float], np.ndarray, dict]:
        """
        Implementa el bucle evolutivo principal según el algoritmo 3.2 del libro.
        En cada generación:
        1. Genera λ descendientes mediante selección de padres, recombinación y mutación
        2. Evalúa los descendientes
        3. Selecciona los μ supervivientes para la siguiente generación

        Args:
            fitness_func: Función de evaluación
            max_gen: Número máximo de generaciones
            recombine_type: Tipo de recombinación a usar
            selection_type: Tipo de selección de supervivientes
        Returns:
            Tuple con:
            - Lista del mejor fitness por generación
            - Mejor vector solución encontrado
            - Diccionario con el histórico completo de la evolución
        """
        best_fitness_history = []
        best_x = None
        history = {'generations': [], 'populations': [], 'timestamps': []}

        population = self.init_population(fitness_func)
        history['populations'].append([{'x': ind.x.copy(), 'sigma': ind.sigma.copy(),
                                        'fitness': ind.fitness} for ind in population])
        history['timestamps'].append(datetime.now())

        for gen in range(max_gen):
            offspring = []
            for _ in range(self.lambda_):
                parents = self.select_parents(population, self.rho)
                child = self.recombine(parents, recombine_type)

                if self.n_sigma == 1:
                    child = self.mutate_one_step(child)
                else:
                    child = self.mutate_n_step(child)

                child.fitness = fitness_func(child.x)
                offspring.append(child)

            population = self.select_survivors(population, offspring, selection_type)

            current_best = min(population, key=lambda ind: ind.fitness)
            best_fitness = current_best.fitness
            best_fitness_history.append(best_fitness)

            if best_x is None or best_fitness < min(best_fitness_history[:-1], default=float('inf')):
                best_x = current_best.x.copy()

            history['generations'].append(gen)
            history['populations'].append([{'x': ind.x.copy(), 'sigma': ind.sigma.copy(),
                                            'fitness': ind.fitness} for ind in population])
            history['timestamps'].append(datetime.now())

            if gen % 100 == 0:
                print(f"Generación {gen}: Mejor fitness = {best_fitness:.6e}")

        return best_fitness_history, best_x, history