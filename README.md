# Dashboard de Perfilacion de Clientes — Credito de Vivienda Colombia

Proyecto de Business Analytics — Universidad de Cundinamarca 2025

Dashboard interactivo para análisis de 28.628 solicitudes de crédito de vivienda en Colombia.

## Tecnologías
- Python + Flask (backend)
- pandas (procesamiento de datos)
- Plotly.js (gráficos interactivos)
- Tailwind CSS (estilos)

## Cómo correr

```bash
pip install -r requirements.txt
python app.py
```

El dashboard abre automáticamente en http://localhost:5000

## Dataset
Adaptado del [Credit Risk Dataset de Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) al contexto colombiano (TRM, ciudades, subsidio Mi Casa Ya, tipos de contrato).

## Características
- 6 filtros interactivos encadenados
- Cross-filtering: clic en gráfica filtra todo el dashboard
- Mapa geográfico de Colombia (Carto Dark Matter)
- Tabla paginada de solicitudes
- Predictores clave de aprobación en tiempo real
