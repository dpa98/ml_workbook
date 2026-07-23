# Proyecto de Clasificación con Perceptrón (Iris Dataset)

Este es un proyecto modularizado de Machine Learning creado a partir del flujo de trabajo estándar en `scikit-learn`. 

## Estructura de Archivos:
- `data_loader.py`: Se encarga de cargar el conjunto de datos Iris y dividirlo en set de entrenamiento y prueba.
- `preprocessing.py`: Realiza el escalado/estandarización de los datos.
- `model.py`: Inicializa, entrena el algoritmo Perceptrón y evalúa su precisión.
- `visualization.py`: Contiene la función para graficar las fronteras de decisión y resaltar el set de test.
- `main.py`: Archivo principal que orquesta todo el flujo de trabajo importando los módulos anteriores.
- `requirements.txt`: Dependencias necesarias.

## Cómo ejecutar:
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar el script: `python main.py`
