# Predicción de Churn en Telecom — README

## Descripción
Proyecto de **clasificación de abandono de clientes (churn)** para una empresa de telecomunicaciones. El flujo está centrado en **pipelines reproducibles** (sin fugas), **balanceo con SMOTE dentro del CV**, **búsqueda de hiperparámetros** optimizando **PR-AUC**, **ajuste de umbral** según criterio de negocio y **análisis por segmentos**.

## Estructura
```
.
├─ data/                     # (opcional) CSV/Parquet con los datos
├─ models/
│  └─ modelo_rf_pipeline.joblib   # Pipeline exportado (OHE+Scaler+SMOTE+RF)
├─ notebooks/
│  ├─ ModeloTelecom2.ipynb        # Notebook original
│  └─ ModeloTelecom2_patched.ipynb# Notebook con sección “A*” (pipeline sin fugas)
└─ README.md
```

## Requisitos
- Python 3.10 o 3.11 (recomendado para compatibilidad estable con scikit-learn/imbalanced-learn)
- Paquetes principales:
  - `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `joblib`

### Instalación rápida (entorno aislado)
```bash
# conda (recomendado)
conda create -n churn-ml python=3.11 -y
conda activate churn-ml
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib joblib

# o con venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib joblib
```

## Flujo principal (Notebook parcheado)
Abrir `notebooks/ModeloTelecom2_patched.ipynb` y ejecutar en orden las celdas:
- **A1–A3**: Imports + Columnas + `Pipeline` con `ColumnTransformer` + `SMOTE` (dentro del CV).
- **A4**: Validación cruzada con métricas para desbalance: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`.
- **A5**: `GridSearchCV` (RF) optimizando **PR-AUC**. Explora `n_estimators`, `max_depth`, `min_samples_*`, `max_features`, `class_weight`.
- **A6**: Hold-out limpio + **ajuste de umbral** (F1 por defecto). Reporte con ROC-AUC, PR-AUC y `classification_report`.
- **A7**: **Análisis por segmento** (ej. `contract`) para priorización operativa.
- **A8**: Exporta pipeline completo a `models/modelo_rf_pipeline.joblib`.

> Nota: No ejecutar las celdas antiguas de `get_dummies`, `MinMaxScaler` y `SMOTE` manual para evitar fugas. El pipeline nuevo se encarga de todo.

## Datos y preprocesamiento
- Target: `churn` (binaria).
- Categóricas: `contract`, `paymentmethod`, `gender`, `internetservice` (ajustable).
- Numéricas: `charges_total`, `charges_monthly`, `tenure` (ajustable).
- Preprocesamiento en `Pipeline`:
  - `OneHotEncoder(handle_unknown='ignore')` para categóricas.
  - `MinMaxScaler` para numéricas.
  - `SMOTE(random_state=42)` **dentro** de la validación cruzada.

## Modelos
- **Random Forest** (principal), optimizado con `GridSearchCV` vía **PR-AUC**.
- **Logistic Regression** baseline con `class_weight='balanced'` (opcional) y opción de calibración.
- Comparación mediante **CV estratificado** (k=5) y métricas enfocadas en desbalance.

## Métricas y evaluación
- **ROC-AUC**: separabilidad general (poco sensible a prevalencia).
- **PR-AUC (Average Precision)**: calidad en clase positiva minoritaria.
- **Ajuste de umbral**: por F1, Youden (TPR–FPR), punto más cercano a (0,1) o por **coste** (FP vs FN).
- Reportes: `classification_report` (precision/recall/F1 por clase) y matriz de confusión.

## Importancia de variables (lectura)
- RF: `feature_importances_` (importancia por reducción de impureza). Suelen destacar:
  - `charges_total`, `charges_monthly`, `tenure`, `contract`, `paymentmethod`, `internetservice`.
- Árbol de Decisión: variables en los **primeros splits**.
- KNN: relevancia implícita vía **distancias** (la **normalización** de numéricas es crítica).

## Salidas clave
- Modelo exportado: `models/modelo_rf_pipeline.joblib` (incluye preprocesamiento completo + RF).
- Métricas: impresiones en A4–A6 (CV y hold-out), curvas ROC/PR si se agregan celdas A9–A11.
- Tabla por segmento (A7) para priorizar campañas de retención.

## Uso en producción (ejemplo)
```python
import joblib
import pandas as pd

pipeline = joblib.load('models/modelo_rf_pipeline.joblib')
X_nuevo = pd.DataFrame([...])  # mismas columnas crudas que entrenamiento (sin OHE/escala manual)
proba = pipeline.predict_proba(X_nuevo)[:,1]
pred  = (proba >= 0.5).astype(int)  # o usar el umbral óptimo encontrado en A6
```

## Mejoras clave frente al flujo anterior
- **Sin fugas**: OHE/Scaler aprenden solo en train por fold; SMOTE dentro del CV.
- **Métricas adecuadas**: incluye **PR-AUC** además de ROC-AUC/Accuracy/F1.
- **Umbral optimizado**: ajustado a F1 o a **coste** FP/FN.
- **Análisis por segmento**: guía acciones de negocio (retención, pricing, bundles).
- **Exportable**: pipeline íntegro con un solo artefacto (`.joblib`).

## Próximos pasos
- Probar **XGBoost/LightGBM** y calibración de probabilidades.
- Ingeniería de variables: `servicios_activos`, `tenure_bins`, `arpu` (cargos mensuales), interacciones.
- Monitoreo en producción: *drift* de datos y recalibración periódica.
