# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       DE/base_algorithm.py
# Descripción:   Base común para los algoritmos evolutivos
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
import random
from common.metrics import ExperimentMetrics
from typing import List, Tuple

class BaseEvolutionaryAlgorithm(ABC):
    """Clase base abstracta con funcionalidad común para algoritmos evolutivos."""

    @property
    @abstractmethod
    def evaluations_per_generation(self) -> int:
        """
        Número de evaluaciones por generación.
        ED: population_size
        EE: lambda_
        """
        pass

    @property
    @abstractmethod
    def seed(self) -> int:
        """Semilla para reproducibilidad."""
        pass

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Tamaño de la población."""
        pass

    # @abstractmethod
    # def evolve(self, fitness_func: callable, max_gen: int):
    #     """
    #     Método abstracto para evolucionar la población.
    #     Debe ser implementado por las clases derivadas.
    #     """
    #     pass

    @abstractmethod
    def evolve(self, fitness_func: callable, max_gen: int,
               recombine_type: str = 'intermediate',
               selection_type: str = 'comma') -> Tuple[List[float], np.ndarray, dict]:
        """
        Método abstracto para evolucionar la población.

        Args:
            fitness_func: Función de evaluación
            max_gen: Número máximo de generaciones
            recombine_type: Tipo de recombinación
            selection_type: Tipo de selección
        Returns:
            Tuple con:
            - Lista del mejor fitness por generación
            - Mejor vector solución encontrado
            - Diccionario con el histórico
        """
        pass


    @staticmethod
    def sphere_function(x: np.ndarray) -> float:
        """
        Función esfera desplazada según sección 2.1 de [Carmona&Galán-2020].
        Función monomodal convexa definida como:
        f(x) = Σ(xi - 10)^2, donde i=1,...,n
        Dominio: [-100,100]^n
        Mínimo global: f(x*) = 0 en x* = (10,...,10)
        """
        return float(np.sum((x - 10.0) ** 2))

    @staticmethod
    def schwefel_function(x: np.ndarray) -> float:
        """
        Función Schwefel según sección 2.2 de [Carmona&Galán-2020].
        Función multimodal no convexa definida como:
        f(x) = 418.9829*n + Σ(-xi*sin(sqrt(|xi|))), donde i=1,...,n
        Dominio: [-500,500]^n
        Mínimo global: f(x*) = 0 en x* = (420.9687,...,420.9687)
        """
        n = len(x)
        return 418.9829 * n + np.sum(-x * np.sin(np.sqrt(np.abs(x))))

    def run_experiment(self, fitness_func: callable, max_gen: int,
                       recombine_type: str = 'intermediate',
                       selection_type: str = 'comma',
                       n_runs: int = 5,
                       epsilon: float = 1e-6) -> ExperimentMetrics:
        """
        Ejecuta múltiples experimentos y recopila métricas según sección 9.2.

        Args:
            fitness_func: Función de evaluación
            max_gen: Número máximo de generaciones
            recombine_type: Tipo de recombinación
            selection_type: Tipo de selección
            n_runs: Número de ejecuciones independientes
            epsilon: Umbral para considerar éxito
        Returns:
            ExperimentMetrics con resultados estadísticos
        """
        if n_runs < 5:
            print("Advertencia: Se recomiendan al menos 5 ejecuciones")

        final_best_values = []
        final_best_vectors = []
        evals_to_success = []
        convergence_curves = []
        all_histories = []
        n_success = 0

        for run in range(n_runs):
            print(f"\nEjecución {run + 1}/{n_runs}...")

            if self.seed is not None:
                run_seed = self.seed + run
                np.random.seed(run_seed)
                random.seed(run_seed)

            history, best_x, run_history = self.evolve(
                fitness_func=fitness_func,
                max_gen=max_gen,
                recombine_type=recombine_type,
                selection_type=selection_type
            )

            convergence_curves.append(history)
            final_fitness = history[-1]
            final_best_values.append(final_fitness)
            final_best_vectors.append(best_x)
            all_histories.append(run_history)

            if final_fitness < epsilon:
                n_success += 1
                for gen, fitness in enumerate(history):
                    if fitness < epsilon:
                        evals_to_success.append(gen * self.evaluations_per_generation)
                        break

        success_rate = float(n_success / n_runs)
        mean_best = float(np.mean(final_best_values))
        std_best = float(np.std(final_best_values))
        mean_evals = float(np.mean(evals_to_success)) if evals_to_success else float('inf')

        return ExperimentMetrics(
            success_rate=success_rate,
            mean_best_fitness=mean_best,
            std_best_fitness=std_best,
            mean_evals_to_success=mean_evals,
            best_vectors=final_best_vectors,
            convergence_curves=convergence_curves,
            full_history={'runs': all_histories}
        )
