# AdalineSGD

ADAptive LInear NEuron (Adaline) classifier trained with stochastic gradient
descent on the Iris dataset. Educational implementation based on
*Python Machine Learning* (Sebastian Raschka).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python scripts/train.py                       # muestra loss curve + frontera de decisión
python scripts/train.py --save out            # guarda la figura como PNG

python scripts/visualize_weights.py           # muestra evolución de pesos
python scripts/visualize_weights.py --save out # guarda la figura como PNG
```

## Estructura

```
adaline_sgd/
├── adaline_sgd/
│   ├── __init__.py    # exports públicos
│   ├── config.py      # configuración centralizada (URL, hiperparámetros)
│   ├── data.py        # carga del dataset y estandarización
│   ├── models.py      # AdalineSGD
│   └── plotting.py    # funciones de visualización
├── scripts/
│   ├── train.py           # pipeline principal (loss + frontera de decisión)
│   └── visualize_weights.py # análisis de evolución de pesos
├── tests/
│   └── test_models.py # tests unitarios (unittest)
└── requirements.txt
```

## Tests

```bash
python -m unittest discover tests
```
