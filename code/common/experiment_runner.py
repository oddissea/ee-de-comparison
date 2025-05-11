# common/experiment_runner.py

"""
Script para ejecutar experimentos de algoritmos evolutivos.
Se centra únicamente en la ejecución y generación de resultados raw.
"""

import os
import sys
from typing import Any, Dict

# Añadir el directorio raíz al path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def get_algorithm_class(algorithm: str):
    """Devuelve la clase del algoritmo según el tipo."""
    if algorithm == "EE":
        from EE.ee import ImprovedEE
        return ImprovedEE
    else:
        from DE.ed import ImprovedDE
        return ImprovedDE


def run_all_configurations(algorithm_instance: Any,
                           fitness_func: callable,
                           func_name: str,
                           max_gen: int,
                           epsilon: float,
                           algorithm: str) -> Dict:
    """
    Ejecuta todas las configuraciones para un algoritmo específico.

    Args:
        algorithm_instance: Instancia del algoritmo (EE o DE)
        fitness_func: Función de fitness a optimizar
        func_name: Nombre de la función (para logs)
        max_gen: Número máximo de generaciones
        epsilon: Criterio de convergencia
        algorithm: Tipo de algoritmo ('EE' o 'DE')

    Returns:
        Dict con los resultados de todas las configuraciones
    """
    print(f"\n{'=' * 20} {func_name} {'=' * 20}")
    print(f"Dimensiones: {algorithm_instance.n_dim}")
    print(f"Dominio: [{algorithm_instance.x_min}, {algorithm_instance.x_max}]^{algorithm_instance.n_dim}")

    results = {}
    config_count = 0

    if algorithm == "EE":
        print(f"Sigmas: {algorithm_instance.n_sigma}")
        print(f"Mutación: {'1-paso' if algorithm_instance.n_sigma == 1 else 'n-pasos'}")

        for recombine_type in ['discrete', 'intermediate']:
            for selection_type in ['comma', 'plus']:
                config_name = f"{recombine_type}-{selection_type}"

                # Establecer la semilla para esta configuración
                if algorithm_instance.seed is not None:
                    algorithm_instance._seed = algorithm_instance.seed + config_count

                print(f"\nConfiguración EE: {config_name} (semilla: {algorithm_instance.seed})")
                metrics = algorithm_instance.run_experiment(
                    fitness_func=fitness_func,
                    max_gen=max_gen,
                    n_runs=5,
                    recombine_type=recombine_type,
                    selection_type=selection_type,
                    epsilon=epsilon,
                )
                results[config_name] = metrics
                print(metrics)
                config_count += 1
    else:  # DE
        for variant in ['rand/1/bin', 'best/1/bin']:
            config_seed = algorithm_instance.seed + config_count if algorithm_instance.seed is not None else None
            algorithm_instance.variant = variant

            print(f"\nConfiguración DE: {variant} (semilla: {config_seed})")
            metrics = algorithm_instance.run_experiment(
                fitness_func=fitness_func,
                max_gen=max_gen,
                n_runs=5,
                epsilon=epsilon
            )
            results[variant] = metrics
            print(metrics)
            config_count += 1

    return results


def setup_directories(algorithm_dir: str):
    """Crear estructura de directorios necesaria."""
    os.makedirs(f"../{algorithm_dir}/resultados/raw", exist_ok=True)


def main(algorithm: str = "EE"):
    """
    Función principal de ejecución.

    Args:
        algorithm: Tipo de algoritmo a ejecutar ('EE' o 'DE')
    """
    _N_DIM = 10
    _POP_SIZE = 30
    _MAX_GEN = 1200
    _EPSILON_SPHERE = 1e-9
    _EPSILON_SCHWEFEL = 1e-3
    _BASE_SEED = 42

    _Algorithm = get_algorithm_class(algorithm)
    setup_directories(algorithm_dir=algorithm)
    results = {}

    if algorithm == "EE":
        _LAMBDA = 210

        # Esfera 1-sigma
        instance_sphere_1 = _Algorithm(
            n_dim=_N_DIM, n_sigma=1, mu=_POP_SIZE, lambda_=_LAMBDA,
            rho=2, x_min=-100, x_max=100, seed=_BASE_SEED
        )
        results['sphere_1sigma'] = run_all_configurations(
            algorithm_instance=instance_sphere_1,
            fitness_func=_Algorithm.sphere_function,
            func_name="Esfera",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SPHERE,
            algorithm=algorithm
        )

        # Esfera n-sigma
        instance_sphere_n = _Algorithm(
            n_dim=_N_DIM, n_sigma=_N_DIM, mu=_POP_SIZE, lambda_=_LAMBDA,
            rho=2, x_min=-100, x_max=100, seed=_BASE_SEED + 1
        )
        results['sphere_n_sigma'] = run_all_configurations(
            algorithm_instance=instance_sphere_n,
            fitness_func=_Algorithm.sphere_function,
            func_name="Esfera",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SPHERE,
            algorithm=algorithm
        )

        # Schwefel 1-sigma
        instance_schwefel_1 = _Algorithm(
            n_dim=_N_DIM, n_sigma=1, mu=_POP_SIZE, lambda_=_LAMBDA,
            rho=5, x_min=-500, x_max=500, seed=_BASE_SEED + 2
        )
        results['schwefel_1sigma'] = run_all_configurations(
            algorithm_instance=instance_schwefel_1,
            fitness_func=_Algorithm.schwefel_function,
            func_name="Schwefel",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SCHWEFEL,
            algorithm=algorithm
        )

        # Schwefel n-sigma
        instance_schwefel_n = _Algorithm(
            n_dim=_N_DIM, n_sigma=_N_DIM, mu=_POP_SIZE, lambda_=_LAMBDA,
            rho=5, x_min=-500, x_max=500, seed=_BASE_SEED + 3
        )
        results['schwefel_n_sigma'] = run_all_configurations(
            algorithm_instance=instance_schwefel_n,
            fitness_func=_Algorithm.schwefel_function,
            func_name="Schwefel",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SCHWEFEL,
            algorithm=algorithm
        )

    else:  # DE
        # Esfera
        instance_sphere = _Algorithm(
            n_dim=_N_DIM, pop_size=_POP_SIZE,
            f=0.5, cr=0.9, x_min=-100, x_max=100,
            seed=_BASE_SEED
        )
        results['sphere'] = run_all_configurations(
            algorithm_instance=instance_sphere,
            fitness_func=_Algorithm.sphere_function,
            func_name="Esfera",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SPHERE,
            algorithm=algorithm
        )

        # Schwefel
        instance_schwefel = _Algorithm(
            n_dim=_N_DIM, pop_size=_POP_SIZE,
            f=0.5, cr=0.9, x_min=-500, x_max=500,
            seed=_BASE_SEED + 1
        )
        results['schwefel'] = run_all_configurations(
            algorithm_instance=instance_schwefel,
            fitness_func=_Algorithm.schwefel_function,
            func_name="Schwefel",
            max_gen=_MAX_GEN,
            epsilon=_EPSILON_SCHWEFEL,
            algorithm=algorithm
        )

    # Guardar solo resultados raw
    from visualization import save_results_to_csv
    for exp_type, exp_results in results.items():
        filename = f"../{algorithm}/resultados/raw/{exp_type}.csv"
        save_results_to_csv(exp_results, filename)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['EE', 'DE']:
        print("Uso: python experiment_runner.py <EE|DE>")
        sys.exit(1)
    main(algorithm=sys.argv[1])

