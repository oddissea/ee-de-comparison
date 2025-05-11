# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       common/analysis.py
# Descripción:   Funciones para analizar resultados de experimentos.
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

import os
import sys
import numpy as np
from visualization import (load_metrics_from_csv, create_convergence_plots,
                           create_convergence_plots_2, create_summary_csv,
                           create_comparison_csv, create_comparative_metrics,
                           save_best_solutions, create_population_distribution_sequence)


def get_experiment_types(algorithm: str) -> list:
    """Devuelve los tipos de experimentos según el algoritmo."""
    if algorithm == "EE":
        return ['sphere_1sigma', 'sphere_n_sigma',
                'schwefel_1sigma', 'schwefel_n_sigma']
    else:  # DE
        return ['sphere', 'schwefel']


def load_all_results(algorithm_dir: str):
    """Carga todos los resultados de un algoritmo específico."""
    results = {}
    for exp_type in get_experiment_types(algorithm_dir):
        results[exp_type] = load_metrics_from_csv(f"../{algorithm_dir}/resultados/raw/{exp_type}.csv")
    return results


def save_results(results: dict, algorithm_dir: str) -> None:
    """Guarda todos los resultados en archivos CSV."""
    # Crear resúmenes individuales
    for exp_type, exp_results in results.items():
        create_summary_csv(exp_results, f"../{algorithm_dir}/resultados/tablas/resumen_{exp_type}.csv")

    # Crear comparativas
    create_comparison_csv(results, f"../{algorithm_dir}/resultados/tablas")


def plot_comparative_results(results: dict,
                             func_name: str,
                             algorithm_dir: str) -> None:
    """Genera gráficas comparativas según el algoritmo."""
    if algorithm_dir == "EE":
        variant1, variant2 = f"{func_name.lower()}_1sigma", f"{func_name.lower()}_n_sigma"
    else:  # DE
        variant1, variant2 = f"{func_name.lower()}", f"{func_name.lower()}"

    curves_data_1 = {
        config_name: data.convergence_curves
        for config_name, data in results[variant1].items()
    }
    curves_data_2 = {
        config_name: data.convergence_curves
        for config_name, data in results[variant2].items()
    }

    generations = np.arange(len(next(iter(curves_data_1.values()))[0]))

    for plot_func in [create_convergence_plots, create_convergence_plots_2]:
        suffix = "_2" if plot_func == create_convergence_plots_2 else ""
        output_file = f"../{algorithm_dir}/resultados/graficas/convergencia{suffix}_{func_name.lower()}.png"

        plot_func(
            generations=generations,
            curves_data_1=curves_data_1,
            curves_data_2=curves_data_2,
            title=f'Comparativa de convergencia - {func_name}',
            output_file=output_file,
            algorithm=algorithm_dir
        )


def main(algorithm_dir: str = "EE"):
    os.makedirs(f"../{algorithm_dir}/resultados/graficas", exist_ok=True)
    os.makedirs(f"../{algorithm_dir}/resultados/tablas", exist_ok=True)
    os.makedirs(f"../{algorithm_dir}/resultados/soluciones", exist_ok=True)

    results = load_all_results(algorithm_dir)

    # Generar gráficas comparativas
    for func_name in ["sphere", "schwefel"]:
        plot_comparative_results(results, func_name, algorithm_dir)

    # Guardar resultados
    save_results(results, algorithm_dir)

    # Guardar mejores soluciones
    for exp_type in get_experiment_types(algorithm_dir):
        save_best_solutions(
            results[exp_type],
            algorithm_dir=algorithm_dir,
            exp_type=exp_type
        )

    # Generar métricas comparativas
    for metric in ['VAMM', 'TE', 'PEX']:
        if algorithm_dir == "EE":
            variants = [('1sigma', 'n_sigma')]
        else:  # DE
            variants = [('rand', 'best')]

        for var1, var2 in variants:
            create_comparative_metrics(
                sphere_results=results[f'sphere{"_" + var1 if algorithm_dir == "EE" else ""}'],
                schwefel_results=results[f'schwefel{"_" + var1 if algorithm_dir == "EE" else ""}'],
                metric=metric,
                algorithm_dir=algorithm_dir)

    # Generar visualizaciones de distribución de población
    output_base = f"../{algorithm_dir}/resultados/distri"
    os.makedirs(output_base, exist_ok=True)

    for exp_type in get_experiment_types(algorithm_dir):
        if algorithm_dir == "EE":
            configurations = [
                'discrete-comma', 'discrete-plus',
                'intermediate-comma', 'intermediate-plus'
            ]
        else:  # DE
            configurations = ['rand/1/bin', 'best/1/bin']

        for config in configurations:
            try:
                history = results[exp_type][config].full_history
                create_population_distribution_sequence(
                    history=history['runs'][0],
                    title=f'{exp_type}_{config}',
                    output_dir=output_base
                )
            except (KeyError, IndexError) as e:
                print(f"Error en {exp_type} - {config}: {str(e)}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['EE', 'DE']:
        print("Uso: python analysis.py <EE|DE>")
        sys.exit(1)
    main(algorithm_dir=sys.argv[1])