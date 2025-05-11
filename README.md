# Estrategias Evolutivas vs Evolución Diferencial
Implementación y análisis comparativo de estrategias evolutivas (EE) y evolución diferencial (ED) para optimización numérica.

## Funciones objetivo
- Función esfera desplazada (monomodal)
- Función Schwefel (multimodal)

## Estructura del proyecto

```text
proyecto/
├── EE/
│   ├── ee.py                    # Implementación de estrategias evolutivas
│   └── resultados/              # Resultados de experimentos con EE
├── DE/
│   ├── ed.py                    # Implementación de evolución diferencial
│   └── resultados/              # Resultados de experimentos con ED
└── common/
    ├── base_algorithm.py        # Clase base para algoritmos evolutivos
    ├── metrics.py               # Métricas de evaluación
    ├── analysis.py              # Análisis de resultados
    ├── experiment_runner.py     # Ejecución de experimentos
    └── visualization.py         # Visualización de resultados
```
# Estrategias Evolutivas vs Evolución Diferencial

Implementación y análisis comparativo de estrategias evolutivas (EE) y evolución diferencial (ED) para optimización numérica.

## Ejecución

1. Ejecutar experimentos:
    ```
    python common/experiment_runner.py <EE|DE>
    ```
2. Analizar resultados:
    ```
    python common/analysis.py <EE|DE>
    ```
Los resultados se almacenarán en:
- `EE/resultados/` o `DE/resultados/` según corresponda
- Incluye tablas CSV con métricas
- Gráficas de convergencia 
- Visualizaciones de la evolución poblacional

## Métricas principales

- TE: Tasa de éxito
- VAMM: Valor de adaptación medio del mejor individuo  
- PEX: Promedio de evaluaciones para alcanzar el éxito

## Configuraciones implementadas

### Estrategias Evolutivas

- Mutación 1-sigma y n-sigma
- Recombinación discreta e intermedia
- Selección (μ,λ) y (μ+λ)

### Evolución Diferencial  

- DE/rand/1/bin
- DE/best/1/bin

## Referencias

Carmona, E. J., & Fernández-Galán, S. (2020). Fundamentos de la Computación Evolutiva. Marcombo.


