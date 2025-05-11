# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       common/visualization.py
# Descripción:   Funciones de visualización y análisis de resultados.
# Versión:       1.0
# Fecha:         08/02/2025
# ------------------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from metrics import ExperimentMetrics
from matplotlib.gridspec import GridSpec

def create_population_distribution_plots(population_data: list,
                                      generation: int,
                                      title: str = None,
                                      output_file: str = None) -> None:
    """
    Genera visualizaciones de la distribución de población según las figuras 5.5-5.13
    de [Carmona&Galán-2020].

    Para ED: Muestra dos gráficas:
    - Distribución de la población sobre las curvas de nivel de la función objetivo
    - Distribución de vectores diferencia centrados en el origen

    Para EE: Muestra una única gráfica con la distribución de la población

    Esta visualización permite analizar:
    - La adaptación de los individuos al paisaje de la función objetivo
    - El progreso de la búsqueda en las cuencas de atracción
    - La distribución y magnitud de los vectores diferencia en ED

    Args:
        population_data: Lista de individuos con sus posiciones y fitness
        generation: Número de generación actual
        title: Título del gráfico (opcional)
        output_file: Ruta para guardar la gráfica (opcional)
    """

    # Extraer posiciones x, y de la población
    x_positions = []
    y_positions = []
    fitness_values = []

    # Extraemos datos directamente de los diccionarios
    for individual in population_data:
        if isinstance(individual, dict) and 'x' in individual:
            x_vector = individual['x']
            x_positions.append(x_vector[0])
            y_positions.append(x_vector[1])
            fitness_values.append(individual['fitness'])

    # Verificar si tenemos datos para graficar
    if not x_positions or not y_positions:
        return

    # Convertir a arrays numpy
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    # Determinar si es EE o ED basándose en el título
    is_ee = '_1sigma' in title or '_n_sigma' in title

    # Crear figura
    if is_ee:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax2 = None
    else:  # ED
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

    # Crear malla para la función objetivo
    if 'schwefel' in title.lower():
        x = np.linspace(-500, 500, 100)
        y = np.linspace(-500, 500, 100)
        _X, _Y = np.meshgrid(x, y)
        _Z = 418.9829 * 2 + (-_X * np.sin(np.sqrt(np.abs(_X)))) + (-_Y * np.sin(np.sqrt(np.abs(_Y))))
        ax1.set_xlim(-500, 500)
        ax1.set_ylim(-500, 500)
        ax1.plot(420.9687, 420.9687, 'rx', markersize=15, markeredgewidth=3,
                 zorder=0)
        if not is_ee:  # Solo para ED
            ax2.set_xlim(-1000, 1000)
            ax2.set_ylim(-1000, 1000)
    else:  # función esfera
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        _X, _Y = np.meshgrid(x, y)
        _Z = (_X - 10)**2 + (_Y - 10)**2
        ax1.set_xlim(-100, 100)
        ax1.set_ylim(-100, 100)
        if not is_ee:  # Solo para ED
            ax2.set_xlim(-200, 200)
            ax2.set_ylim(-200, 200)

    # Añadir contornos de la función
    ax1.contour(_X, _Y, _Z, levels=30, cmap='viridis', alpha=0.5)

    # Plot 1: Distribución de población
    scatter = ax1.scatter(x_positions, y_positions,
                          c="blue",
                          s=50, alpha=0.8)
    fig.colorbar(scatter, ax=ax1, label='Fitness')

    # Configurar etiquetas
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.grid(True)

    # Para ED, añadir el plot de vectores diferencia
    if not is_ee:
        diff_x = []
        diff_y = []
        for i in range(len(x_positions)):
            for j in range(i + 1, len(x_positions)):
                diff_x.append(x_positions[j] - x_positions[i])
                diff_y.append(y_positions[j] - y_positions[i])

        if diff_x and diff_y:
            ax2.scatter(diff_x, diff_y, c='blue', s=20, alpha=0.3)
            ax2.set_xlabel('$\Delta x_1$')
            ax2.set_ylabel('$\Delta x_2$')
            ax2.grid(True)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Configurar títulos
    if title:
        if is_ee:
            plt.title(f"{title} (generación {generation})")
        else:
            plt.suptitle(f"{title} (generación {generation})")
            ax1.set_title('Distribución de población')
            ax2.set_title('Distribución de vectores diferencia')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_population_distribution_sequence(history: dict,
                                            title: str = None,
                                            output_dir: str = None,
                                            generations: list = None) -> None:
    """
    Genera una secuencia de gráficas mostrando la evolución de la población según la
    sección 5.12 de [Carmona&Galán-2020].

    Produce visualizaciones para generaciones específicas que permiten analizar:
    - El proceso de exploración inicial del espacio de búsqueda
    - La identificación y convergencia hacia las cuencas de atracción
    - La explotación local dentro de las cuencas prometedoras
    - La adaptación de los vectores diferencia durante la evolución

    Args:
        history: Diccionario con el histórico de poblaciones
        title: Título base para las gráficas (opcional)
        output_dir: Directorio para guardar la secuencia (opcional)
        generations: Lista de generaciones a visualizar (opcional)
    """

    if generations is None:
        generations = [1, 2, 4, 6, 8, 100, 500, 1000, 1200]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Accedemos directamente a populations
    populations = history.get('populations', [])

    if not populations:
        # print("No se encontraron datos de población")
        return

    for gen in generations:
        population_data = populations[gen]

        output_file = None
        if output_dir:
            output_file = os.path.join(
                output_dir,
                f"{title.replace('/', '_')}_gen_{gen:03d}.png"
            )

        create_population_distribution_plots(
            population_data=population_data,
            generation=gen,
            title=f"{title} - Generación {gen}",
            output_file=output_file
        )

def _process_curves_data(curves: list, generations: np.ndarray, color: str,
                         label: str, ax: plt.Axes, use_semi_logy: bool = False,
                         error_config: dict = None,
                         title_contains_schwefel: bool = False) -> None:
    curves_array = np.array(curves)
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    best_curve = np.min(curves_array, axis=0)

    if use_semi_logy:
        ax.semilogy(generations, mean_curve, color=color,
                   linewidth=2, label=f"{label} (media)")
        if title_contains_schwefel:
            ax.semilogy(generations, best_curve, color=color,
                       linewidth=1, linestyle='--',
                       label=f"{label} (mejor)")
    else:
        ax.plot(generations, mean_curve, color=color,
                linewidth=2, label=f"{label} (media)")
        if title_contains_schwefel:
            ax.plot(generations, best_curve, color=color,
                    linewidth=1, linestyle='--',
                    label=f"{label} (mejor)")

    ax.fill_between(generations,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color, alpha=0.2)

    if error_config:
        ax.errorbar(generations[error_config['indices']],
                    mean_curve[error_config['indices']],
                    yerr=std_curve[error_config['indices']],
                    fmt='none',
                    color=color, alpha=0.3, capsize=3)


def _setup_plot_axes(ax1: plt.Axes, ax2: plt.Axes, title: str,
                     algorithm: str = 'EE',
                     use_log_scale: bool = False,
                     y_max: int = None,
                     my_ticks: int = 500) -> None:
    """Configura los ejes de las gráficas."""
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlabel('Generación')
        ax.set_ylabel('Mejor fitness' + (' (escala log)' if use_log_scale else ''))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if y_max is not None:
            ax.set_ylim(-100, y_max)
            ax.set_yticks(np.arange(0, y_max + 100, my_ticks))
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.1)

    if algorithm == 'EE':
        ax1.set_title('Mutación 1-sigma')
        ax2.set_title('Mutación n-sigma')
    else:  # DE
        ax1.set_title('Variante rand/1/bin')
        ax2.set_title('Variante best/1/bin')

    plt.suptitle(title)
    plt.tight_layout()


def create_convergence_plots(generations: np.ndarray,
                           curves_data_1: dict,
                           curves_data_2: dict,
                           title: str,
                           output_file: str = None,
                           y_max: int = 4000,
                           algorithm: str = "EE") -> None:
    """
    Genera curvas de progreso según la sección 9.5.1 de [Carmona&Galán-2020].

    Produce dos gráficas comparativas mostrando:
    - Evolución del mejor individuo
    - Media y desviación estándar de la población
    - Escala logarítmica para mejor visualización

    La comparación permite analizar:
    - Velocidad de convergencia
    - Estabilidad del proceso evolutivo
    - Consistencia entre ejecuciones

    Args:
        generations: Array con números de generación
        curves_data_1: Datos de convergencia del primer conjunto
        curves_data_2: Datos de convergencia del segundo conjunto
        title: Título del gráfico
        output_file: Ruta para guardar la gráfica (opcional)
        y_max: Valor máximo para el eje y
        algorithm: Tipo de algoritmo ('EE' o 'DE')
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (label, curves), color in zip(curves_data_1.items(), colors):
        _process_curves_data(curves, generations, color, label, ax1,
                             title_contains_schwefel="Schwefel" in title)

    for (label, curves), color in zip(curves_data_2.items(), colors):
        _process_curves_data(curves, generations, color, label, ax2,
                             title_contains_schwefel="Schwefel" in title)

    _setup_plot_axes(ax1, ax2, title, algorithm=algorithm, y_max=y_max)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _create_error_config(i: int, total_gens: int, n_points: int, n_items: int) -> dict:
    """Crea la configuración de error para las barras de error."""
    return {
        'indices': range(
            (i * total_gens // n_points) // n_items,
            total_gens - total_gens // n_points,
            total_gens // n_points
        )
    }

def _plot_curves_data(curves_data: dict, ax: plt.Axes, generations: np.ndarray,
                     colors: list, total_gens: int, n_points: int, title: str) -> None:
    """Dibuja las curvas de datos para un eje dado."""
    for i, ((label, curves), color) in enumerate(zip(curves_data.items(), colors)):
        error_config = _create_error_config(i, total_gens, n_points, len(curves_data))
        _process_curves_data(curves, generations, color, label, ax,
                             use_semi_logy=True,
                             error_config=error_config,
                             title_contains_schwefel="Schwefel" in title)


def create_convergence_plots_2(generations: np.ndarray,
                             curves_data_1: dict,
                             curves_data_2: dict,
                             title: str,
                             output_file: str = None,
                             algorithm: str = "EE") -> None:
    """
    Genera curvas de convergencia alternativas según la sección 9.5.1 de
    [Carmona&Galán-2020].

    Variante que incluye:
    - Escala logarítmica para mejor visualización de la convergencia
    - Barras de error en puntos específicos para análisis estadístico
    - Curvas de mejor individuo absoluto junto a la media

    Esta visualización complementa a create_convergence_plots() permitiendo
    un análisis más detallado de la estabilidad y consistencia del proceso
    evolutivo.

    Args:
        generations: Array con números de generación
        curves_data_1: Datos de convergencia del primer conjunto
        curves_data_2: Datos de convergencia del segundo conjunto
        title: Título del gráfico
        output_file: Ruta para guardar la gráfica (opcional)
        algorithm: Tipo de algoritmo ('EE' o 'DE')
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    _N_POINTS = 10
    total_gens = len(generations)

    _plot_curves_data(curves_data_1, ax1, generations, colors, total_gens, _N_POINTS, title)
    _plot_curves_data(curves_data_2, ax2, generations, colors, total_gens, _N_POINTS, title)

    _setup_plot_axes(ax1, ax2, title, algorithm=algorithm, use_log_scale=True)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparative_metrics(sphere_results: dict,
                           schwefel_results: dict,
                           metric: str, algorithm_dir: str = "EE") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    """
    Genera gráficas comparativas de métricas según la sección 9.4.2 de 
    [Carmona&Galán-2020].

    Permite contrastar el comportamiento del algoritmo en las funciones 
    esfera y Schwefel para una métrica específica (TE, VAMM o PEX).
    La comparación ayuda a evaluar:
    - Eficacia del algoritmo en diferentes tipos de problemas
    - Diferencias estadísticamente significativas entre configuraciones
    - Robustez frente a cambios en la función objetivo

    Args:
        sphere_results: Resultados para la función esfera
        schwefel_results: Resultados para la función Schwefel
        metric: Métrica a comparar ('TE', 'VAMM' o 'PEX')
        algorithm_dir: Tipo de algoritmo ('EE' o 'DE')
    """

    metric_map = {
        'VAMM': 'mean_best_fitness',
        'TE': 'success_rate',
        'PEX': 'mean_evals_to_success'
    }

    configs = list(sphere_results.keys())
    sphere_values = _process_metric_data(sphere_results, metric, metric_map)
    ax1.bar(configs, sphere_values)
    ax1.set_title(f'{metric} - Esfera')
    ax1.tick_params(axis='x', rotation=45)

    schwefel_values = _process_metric_data(schwefel_results, metric, metric_map)
    ax2.bar(configs, schwefel_values)
    ax2.set_title(f'{metric} - Schwefel')
    ax2.tick_params(axis='x', rotation=45)

    if metric == 'PEX':
        plt.figtext(0.5, 0.01,
                   'Nota: Se muestra 0 para configuraciones sin éxito (PEX = inf)',
                   ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"../{algorithm_dir}/resultados/graficas/comparativa_{metric.lower()}.png")
    plt.close()


def save_best_solutions(results: dict, algorithm_dir: str, exp_type: str):
    """
    Guarda las mejores soluciones encontradas.

    Args:
        results: Diccionario con los resultados
        algorithm_dir: Directorio del algoritmo ('EE' o 'DE')
        exp_type: Tipo de experimento
    """
    csv_file = f"../{algorithm_dir}/resultados/soluciones/mejores_{exp_type}.csv"
    txt_file = f"../{algorithm_dir}/resultados/soluciones/mejores_{exp_type}.txt"

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuración', 'Mejor_Fitness', 'Vector_Solución'])

        for config, metrics in results.items():
            best_fitness = float('inf')
            best_vector = None

            for i, curve in enumerate(metrics.convergence_curves):
                if min(curve) < best_fitness:
                    best_fitness = min(curve)
                    best_vector = metrics.best_vectors[i]

            writer.writerow([config, f"{best_fitness:.6e}",
                             json.dumps(best_vector.tolist())])

    with open(txt_file, 'w') as f:
        for config, metrics in results.items():
            best_fitness = float('inf')
            best_vector = None

            for i, curve in enumerate(metrics.convergence_curves):
                if min(curve) < best_fitness:
                    best_fitness = min(curve)
                    best_vector = metrics.best_vectors[i]

            f.write(f"\nConfiguración: {config}\n")
            f.write(f"Mejor fitness: {best_fitness:.6e}\n")
            f.write(f"Vector solución: {best_vector}\n")
            f.write("-" * 50)

def _process_metric_data(results: dict, metric: str, metric_map: dict) -> list:
    """
    Función auxiliar para procesar datos de métricas según la sección 9.2 de
    [Carmona&Galán-2020].

    Extrae y procesa los valores de una métrica específica para todas las
    configuraciones. Gestiona casos especiales como:
    - PEX infinito cuando no hay éxitos
    - Normalización de valores según la métrica

    Args:
        results: Diccionario con resultados por configuración
        metric: Métrica a procesar ('TE', 'VAMM' o 'PEX')
        metric_map: Mapeo entre nombres de métricas y atributos
    Returns:
        Lista con los valores procesados de la métrica
    """

    values = []
    for config in results.keys():
        val = getattr(results[config], metric_map[metric])
        if metric == 'PEX' and np.isinf(val):
            values.append(0)
        else:
            values.append(val)
    return values


def save_results_to_csv(results: dict, filename: str) -> None:
    """
    Guarda los resultados siguiendo las recomendaciones de la sección 9.4 de
    [Carmona&Galán-2020] para permitir análisis estadístico posterior.

    Almacena:
    - Métricas principales (TE, VAMM, PEX)
    - Vectores solución
    - Curvas de convergencia
    - Histórico completo de la evolución

    Args:
        results: Diccionario con los resultados del experimento
        filename: Ruta del archivo CSV
    """

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ["config_name", "success_rate", "mean_best_fitness",
                   "std_best_fitness", "mean_evals_to_success",
                   "best_vectors", "convergence_curves", "seed", "full_history"]
        writer.writerow(headers)

        for config_name, metrics in results.items():
            # Vectores solución y curvas de convergencia
            best_vectors_str = json.dumps([vec.tolist() for vec in metrics.best_vectors])
            convergence_curves_str = json.dumps(metrics.convergence_curves)

            # Procesamos el histórico completo
            history_dict = {'runs': []}
            for run in metrics.full_history.get('runs', []):
                run_dict = {'populations': []}
                # Por cada generación
                for population in run.get('populations', []):
                    generation = []
                    # Por cada individuo en la población
                    for ind in population:
                        # Convertimos el array numpy a lista para serialización
                        ind_dict = {
                            'x': ind['x'].tolist() if isinstance(ind['x'], np.ndarray) else ind['x'],
                            'fitness': float(ind['fitness'])
                        }
                        generation.append(ind_dict)
                    run_dict['populations'].append(generation)
                history_dict['runs'].append(run_dict)

            history_str = json.dumps(history_dict)

            writer.writerow([
                config_name,
                metrics.success_rate,
                metrics.mean_best_fitness,
                metrics.std_best_fitness,
                metrics.mean_evals_to_success,
                best_vectors_str,
                convergence_curves_str,
                metrics.seed,
                history_str
            ])


def load_metrics_from_csv(filename: str) -> dict:
    """
    Carga las métricas desde un archivo CSV según las recomendaciones de la
    sección 9.4 de [Carmona&Galán-2020].

    Recupera los datos necesarios para realizar análisis estadístico:
    - Índices de prestaciones (TE, VAMM, PEX)
    - Vectores solución y curvas de convergencia
    - Histórico completo de las ejecuciones

    Args:
        filename: Ruta del archivo CSV con los resultados
    Returns:
        Diccionario con las métricas procesadas por configuración
    """

    # Aumentar el límite de tamaño de campo
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    # print(f"\nIniciando carga desde: {filename}")
    data_dict = {}

    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        # print("Cabeceras encontradas:", headers)
        idx = {name: headers.index(name) for name in headers}

        for row in reader:
            config_name = row[idx["config_name"]]
            history_raw = row[idx["full_history"]]

            best_vectors = [np.array(vec) for vec in json.loads(row[idx["best_vectors"]])]
            convergence_curves = json.loads(row[idx["convergence_curves"]])
            full_history = json.loads(history_raw)
            if 'runs' in full_history:
                for run_idx, run in enumerate(full_history['runs']):
                    if 'populations' in run:
                        processed_populations = []
                        for gen_idx, generation in enumerate(run['populations']):
                            processed_generation = []

                            if isinstance(generation, dict):
                                ind = generation  # Si es un diccionario, es un solo individuo
                                generation = [ind]  # Lo convertimos en lista

                            for ind_idx, ind in enumerate(generation):
                                if isinstance(ind, dict) and 'x' in ind and 'fitness' in ind:
                                    ind_processed = {
                                        'x': np.array(ind['x']),
                                        'fitness': float(ind['fitness'])
                                    }
                                    processed_generation.append(ind_processed)
                            processed_populations.append(processed_generation)
                        run['populations'] = processed_populations
            data_dict[config_name] = ExperimentMetrics(
                success_rate=float(row[idx["success_rate"]]),
                mean_best_fitness=float(row[idx["mean_best_fitness"]]),
                std_best_fitness=float(row[idx["std_best_fitness"]]),
                mean_evals_to_success=float(row[idx["mean_evals_to_success"]]),
                best_vectors=best_vectors,
                convergence_curves=convergence_curves,
                seed=int(row[idx["seed"]]) if "seed" in idx and row[idx["seed"]].strip() else None,
                full_history=full_history
            )
    return data_dict

def create_summary_csv(results: dict, output_file: str) -> None:
    """
    Genera un resumen CSV con los índices de prestaciones según la sección 9.2
    de [Carmona&Galán-2020].

    El resumen incluye por cada configuración:
    - Tasa de éxito (TE)
    - Valor de adaptación medio del mejor individuo (VAMM)
    - Promedio de evaluaciones para alcanzar el éxito (PEX)

    Args:
        results: Diccionario con los resultados del experimento
        output_file: Ruta donde guardar el archivo CSV
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuración', 'TE', 'VAMM', 'PEX'])

        for config, metrics in results.items():
            writer.writerow([
                config,
                f"{metrics.success_rate:.2%}",
                f"{metrics.mean_best_fitness:.6e}",
                f"{metrics.mean_evals_to_success:.2f}"
            ])


def create_comparison_csv(results: dict, output_dir: str) -> None:
    """
    Crea archivos CSV con comparativas.

    Args:
        results: Diccionario con resultados
        output_dir: Directorio de salida (incluye el tipo de algoritmo en la ruta)
    """
    metrics = {
        'TE': 'success_rate',
        'VAMM': 'mean_best_fitness',
        'PEX': 'mean_evals_to_success'
    }

    # Resúmenes individuales
    for exp_type, exp_results in results.items():
        create_summary_csv(exp_results, f"{output_dir}/resumen_{exp_type}.csv")

    # Determinar el formato según el algoritmo
    is_ee = "sphere_1sigma" in results

    # Comparativas por métrica
    for metric_name, metric_key in metrics.items():
        output_file = f"{output_dir}/comparativa_{metric_name.lower()}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Función', 'Configuración', metric_name])

            if is_ee:
                # Formato para EE
                for func_type in ['sphere', 'schwefel']:
                    for sigma_type in ['1sigma', 'n_sigma']:
                        key = f"{func_type}_{sigma_type}"
                        for config, data in results[key].items():
                            writer.writerow([
                                func_type.capitalize(),
                                f"{sigma_type}-{config}",
                                f"{getattr(data, metric_key):.6e}"
                            ])
            else:
                # Formato para DE
                for func_type in ['sphere', 'schwefel']:
                    for config, data in results[func_type].items():
                        writer.writerow([
                            func_type.capitalize(),
                            config,
                            f"{getattr(data, metric_key):.6e}"
                        ])