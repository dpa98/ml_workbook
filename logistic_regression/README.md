# Proyecto de Regresión Logística Modular

Este repositorio implementa un clasificador lineal probabilístico (Regresión Logística) sobre el conjunto de datos Iris. 
A diferencia de los flujos basados en notebooks monolíticos, este proyecto está dividido en scripts limpios e independientes, orquestados a través de una interfaz de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `LogisticRegression` y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la terminal sin necesidad de abrir y editar el código.

Ejecución estándar:
`python main.py`

Modificando el parámetro de regularización C y el tamaño del set de prueba:
`python main.py --c_param 0.1 --test_size 0.4`
`python main.py --c_param 100.0`
