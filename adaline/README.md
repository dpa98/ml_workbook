# Adaline

Implementación educativa del clasificador **Adaline** (*ADAptive LInear
NEuron*) entrenado con descenso de gradiente sobre el dataset Iris, basada en
*Python Machine Learning* (Sebastian Raschka).

## Qué es Adaline

Adaline es un modelo lineal binario (clases 0/1) que aprende los pesos `w` y
el sesgo `b` minimizando la **función de pérdida de mínimos cuadrados** (MSE)
mediante descenso de gradiente *batch*:

- **Salida lineal**: `z = w·x + b`
- **Pérdida**: `L = mean((y - z)²)`
- **Actualización por época** (todo el dataset a la vez):

  ```
  w += eta · 2 · Xᵀ·(y − z) / n
  b += eta · 2 · mean(y − z)
  ```

- **Predicción**: umbral sobre la salida lineal (`z ≥ 0.5 → clase 1`)

La convergencia depende del *learning rate* (`eta`) y de la escala de las
características: por eso el pipeline estandariza los datos (media 0,
varianza 1) antes de entrenar. Con `eta=0.5` la pérdida cae de ~0.49 a ~0.01
en las primeras ~10 épocas.

> **Nota de compatibilidad**: la implementación (`models.py`) corresponde a
> la 3ª edición del libro (*Machine Learning with PyTorch and Scikit-Learn*),
> que normaliza la pérdida por el número de muestras. Los hiperparámetros de
> `Config` (`eta=0.5`, `n_iter=50`) están ajustados para que la curva de
> pérdida coincida con la figura de convergencia de la 1ª edición
> (*Python Machine Learning*), cuyo gradiente no está normalizado.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python scripts/train.py               # entrena y muestra los gráficos
python scripts/train.py --save out    # guarda los gráficos como PNG en out/
python scripts/weights.py             # muestra la evolución de los pesos por época
python scripts/weights.py --save out  # guarda el gráfico como PNG
```

### `scripts/train.py`

Entrena `AdalineGD` sobre setosa/versicolor y muestra dos paneles:

1. **Regiones de decisión** sobre las features estandarizadas (sepal length
   y petal length), con las muestras de cada clase.
2. **Curva de pérdida (MSE) por época**.

### `scripts/weights.py`

Replica el bucle de entrenamiento registrando `w1`, `w2` y `b` en cada época,
y muestra cómo convergen a sus valores finales (~`w1=2.3`, `w2=4.7`, `b=-1.2`)
en las primeras ~10 épocas.

## Configuración

Todos los hiperparámetros están centralizados en `adaline/config.py`
(dataclass `Config`):

| Campo | Valor | Descripción |
|---|---|---|
| `data_url` | seaborn-data `iris.csv` | Origen del dataset |
| `local_path` | `None` | Ruta local alternativa (usa `data_url` si es `None`) |
| `target_col` | `species` | Columna de etiquetas |
| `feature_cols` | `sepal_length`, `petal_length` | Features del modelo |
| `class_map` | `setosa→0`, `versicolor→1` | Codificación binaria |
| `eta` | `0.5` | Learning rate |
| `n_iter` | `50` | Épocas de entrenamiento |
| `random_state` | `1` | Semilla de inicialización de pesos |

Los scripts leen los hiperparámetros desde `Config`, así que cambiarlos ahí
afecta a todos los scripts y tests por igual.

## Estructura

```
adaline/
├── adaline/
│   ├── __init__.py    # exports públicos
│   ├── config.py      # configuración centralizada (URL, hiperparámetros)
│   ├── data.py        # carga del dataset y estandarización
│   ├── models.py      # AdalineGD (descenso de gradiente)
│   └── plotting.py    # funciones de visualización
├── scripts/
│   ├── train.py       # orquestación del pipeline (regiones + pérdida)
│   └── weights.py     # evolución de los pesos por época
├── tests/
│   └── test_models.py # tests unitarios (unittest)
└── requirements.txt   # numpy, pandas, matplotlib
```

## Tests

```bash
python -m unittest discover tests
```

Cubren convergencia sobre datos separables, precisión ≥ 95% en el pipeline de
Iris, reproducibilidad con `random_state` e inicialización de pesos, además
de la estandarización (media 0, varianza 1, sin mutar la entrada).
