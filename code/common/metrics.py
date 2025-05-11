# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       common/metrics.py
# Descripción:   Métricas para evaluar el rendimiento del algoritmo
#                según [Carmona&Galán-2020].
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class ExperimentMetrics:
    """
    Implementa las métricas de evaluación según [Carmona&Galán-2020].

    Calcula y almacena los tres índices principales de medida de prestaciones:
    - Tasa de éxito (TE): Sección 9.2.1 - Cociente entre ejecuciones exitosas y total
    - Valor de adaptación medio del mejor individuo (VAMM): Sección 9.2.2
    - Promedio de evaluaciones para alcanzar el éxito (PEX): Sección 9.2.3

    También mantiene el histórico de convergencia y vectores solución para análisis.

    Atributos:
        success_rate (float): Tasa de éxito (TE) en [0,1]
        mean_best_fitness (float): VAMM - Media del mejor fitness en cada ejecución
        std_best_fitness (float): Desviación estándar del mejor fitness
        mean_evals_to_success (float): PEX - Media de evaluaciones hasta éxito
        best_vectors (List[np.ndarray]): Mejores vectores solución de cada ejecución
        convergence_curves (List[List[float]]): Curvas de convergencia según sección 9.5.1
        timestamps (List[datetime]): Registro temporal de ejecución
        full_history (Dict): Histórico completo para análisis
        seed (int): Semilla aleatoria usada
    """
    success_rate: float
    mean_best_fitness: float
    std_best_fitness: float
    mean_evals_to_success: float
    best_vectors: List[np.ndarray]
    convergence_curves: List[List[float]]
    timestamps: List[datetime] = field(default_factory=list)
    full_history: Dict = field(default_factory=dict)
    seed: int = None

    def __str__(self) -> str:
        return (
            f"Tasa de éxito (TE): {self.success_rate:.2%}\n"
            f"VAMM: {self.mean_best_fitness:.6e} ± {self.std_best_fitness:.6e}\n"
            f"PEX: {self.mean_evals_to_success:.2f}"
        )

    def plot_convergence(self, title: str = "Curva de convergencia"):
        plt.figure(figsize=(10, 6))
        curves = np.array(self.convergence_curves)
        generations = np.arange(curves.shape[1])

        for i in range(curves.shape[0]):
            plt.semilogy(generations, curves[i], alpha=0.3,
                         label=f'Ejecución {i + 1}')

        mean_curve = np.mean(curves, axis=0)
        plt.semilogy(generations, mean_curve, 'r-', linewidth=2,
                     label='Media')

        plt.grid(True)
        plt.xlabel('Generación')
        plt.ylabel('Mejor fitness')
        plt.title(title)
        plt.legend()
        plt.show()

    def get_full_history(self) -> Dict:
        """Retorna el histórico completo del experimento."""
        return {
            'timestamp': [t.isoformat() for t in self.timestamps],
            'best_vectors': [vec.tolist() for vec in self.best_vectors],
            'convergence_curves': self.convergence_curves,
            'metrics': {
                'success_rate': self.success_rate,
                'mean_best_fitness': self.mean_best_fitness,
                'std_best_fitness': self.std_best_fitness,
                'mean_evals_to_success': self.mean_evals_to_success
            }
        }

    def get_best_solution(self) -> tuple:
        """Retorna el mejor vector solución y su fitness."""
        best_idx = np.argmin([curve[-1] for curve in self.convergence_curves])
        return self.best_vectors[best_idx], self.convergence_curves[best_idx][-1]

    def add_run_data(self, best_vector: np.ndarray, convergence_curve: List[float]):
        """Añade datos de una nueva ejecución."""
        self.best_vectors.append(best_vector)
        self.convergence_curves.append(convergence_curve)
        self.timestamps.append(datetime.now())