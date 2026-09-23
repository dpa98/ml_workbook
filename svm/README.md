# Proyecto de Máquinas de Vectores de Soporte (SVM) Modular

Este repositorio implementa un clasificador de máquinas de vectores de soporte sobre el conjunto de datos Iris,
replicando la estructura de los proyectos knn y perceptrón pero cambiando la lógica del modelo. Al igual que los
demás flujos de tensorhub, está dividido en scripts limpios e independientes orquestados a través de una interfaz
de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `SVC` y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la terminal
sin necesidad de abrir y editar el código.

Ejecución estándar:
`python main.py`

Modificando el parámetro de regularización C y el kernel:
`python main.py --c_param 100.0 --kernel linear`
`python main.py --c_param 0.1 --kernel rbf --test_size 0.4`