# Proyecto de SVM con kernel RBF y gamma pequeño

Este repositorio implementa una Máquina de Vectores de Soporte (SVM) con kernel RBF sobre el conjunto de
datos Iris, usando un valor pequeño de gamma (0.1 por defecto) para obtener una frontera de decisión
suave. Replica la estructura de los proyectos knn, svm y perceptrón, orquestados a través de una
interfaz de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `SVC` con kernel rbf y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la
terminal sin necesidad de abrir y editar el código.

Ejecución estándar (gamma=0.1):
`python main.py`

Modificando gamma y el resto de hiperparámetros:
`python main.py --gamma 0.01`
`python main.py --c_param 100.0 --gamma 0.1 --test_size 0.4`