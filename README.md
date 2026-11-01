# 🎓 EduPredict AI

**Sistema inteligente de análisis académico y predicción de deserción universitaria mediante algoritmos clásicos de Machine Learning.**

Proyecto ACA N.º 1 — Aprendizaje Automático · Especialización en Inteligencia Artificial · Corporación Unificada Nacional de Educación Superior (CUN).

![Dashboard de EduPredict AI](docs/capturas/dashboard.png)

---

## 1. Descripción

EduPredict AI analiza el histórico de 4.424 estudiantes de educación superior, entrena y compara tres clasificadores clásicos de Scikit-learn y presenta los resultados en una aplicación Streamlit con cuatro secciones: **Dashboard**, **Predicción individual**, **Comparación de modelos** y **Acerca del proyecto**.

Las predicciones provienen exclusivamente de pipelines de Scikit-learn entrenados localmente; no se usan APIs de IA generativa.

## 2. Situación problema

La deserción universitaria tiene costos académicos, económicos y sociales. Las instituciones suelen identificar a los estudiantes en riesgo cuando ya es tarde para intervenir.

**Pregunta orientadora:** ¿Cómo transformar datos educativos en conocimiento predictivo para anticipar la deserción de un estudiante y facilitar la identificación de posibles situaciones de riesgo académico?

## 3. Objetivo general

Desarrollar una solución de Machine Learning clásico que permita analizar información académica, entrenar y comparar modelos de clasificación, evaluar su desempeño y presentar predicciones mediante una aplicación interactiva de Streamlit.

## 4. Dataset

| Atributo | Valor |
|---|---|
| Nombre | *Predict Students' Dropout and Academic Success* |
| Fuente | [UCI Machine Learning Repository, ID 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89) |
| Registros | 4.424 estudiantes (sin nulos ni duplicados) |
| Variables predictoras | 36: 9 nominales, 8 binarias y 19 numéricas |
| Objetivo | 3 clases al final de la duración normal del programa: **Dropout** (Deserción, 32,1 %), **Enrolled** (Matriculado, 17,9 %), **Graduate** (Graduado, 49,9 %) |
| Archivo versionado | `data/data.csv` (CSV original de UCI, separador `;`, SHA-256 `3ef126de…c45ae`) |
| Archivo limpio | `data/estudiantes_limpio.csv` (nombres de columna normalizados, tipos validados) |

**Diccionario de datos.** Las columnas, unidades, rangos y la traducción al español de **todos** los códigos categóricos (programa, modalidad de admisión, formación previa, nacionalidad, nivel educativo y ocupación de los padres, etc.) están en [`src/data_dictionary.py`](src/data_dictionary.py) y se muestran en la aplicación (*Acerca del proyecto → Diccionario de variables*). Resumen de unidades:

| Grupo | Variables | Unidad |
|---|---|---|
| Notas de ingreso | Nota de admisión, nota de formación previa | escala 0–200 |
| Rendimiento semestral | Promedio de notas del 1.er / 2.º semestre | escala 0–20 (0 = ninguna unidad aprobada) |
| Unidades curriculares | Inscritas, aprobadas, convalidadas, evaluaciones, sin evaluación | número de asignaturas/evaluaciones |
| Personales | Edad al matricularse | años |
| Macroeconómicas | Desempleo, inflación (%), PIB (variación) | del año de ingreso, Portugal |
| Binarias | Becario, deudor, matrícula al día, desplazado, internacional, NEE, sexo, jornada | 1 = sí / 0 = no (sexo: 1 = masculino; jornada: 1 = diurna) |

El dataset se puede obtener de dos formas: (a) usar el CSV versionado en `data/data.csv` (la aplicación nunca necesita internet), o (b) descargarlo de UCI con `python -m src.train --download` (descarga solo si falta).

## 5. Arquitectura

```
edupredict-ml-streamlit/
├── app.py                         # Aplicación Streamlit (no entrena; solo carga artefactos)
├── requirements.txt               # Dependencias de ejecución (versiones fijadas)
├── requirements-dev.txt           # + pruebas y notebook
├── pytest.ini
├── .streamlit/config.toml         # Tema visual
├── data/
│   ├── data.csv                   # CSV original UCI 697
│   └── estudiantes_limpio.csv     # Resultado de src/data_loader.py
├── notebooks/
│   └── ACA1_EDA_y_Modelado.ipynb  # EDA, limpieza y resultados (ejecutado)
├── src/
│   ├── data_dictionary.py         # Columnas, tipos, unidades, códigos → español
│   ├── data_loader.py             # Descarga, lectura, limpieza y validación de consistencia
│   ├── preprocessing.py           # Conjuntos de variables, pipelines por modelo, validación de esquema
│   ├── train.py                   # GridSearchCV + selección por CV + exportación de artefactos
│   ├── evaluate.py                # Métricas multiclase, importancia por permutación
│   ├── predict.py                 # Carga de artefactos y predicción (usado por la app y las pruebas)
│   └── project_info.py            # Textos académicos, integrantes y enlaces
├── models/
│   ├── modelo.joblib / metadata.json / metricas.json                         # Información completa
│   └── modelo_temprano.joblib / metadata_temprano.json / metricas_temprano.json  # Alerta temprana
├── scripts/build_poster.py        # Genera docs/poster/poster.html desde las métricas reales
├── tests/                         # pytest: preprocesamiento, predicción y arranque de Streamlit
└── docs/
    ├── capturas/                  # Capturas reales de la aplicación
    ├── poster/                    # Contenido base y borrador del póster
    ├── referencias/
    └── video/
```

Flujo: `data.csv → data_loader (limpieza determinista) → train (split 80/20 → GridSearchCV 5-fold → selección por CV → evaluación en prueba) → models/*.joblib + *.json → predict → app.py`.

> Se agregaron `data_dictionary.py` y `project_info.py` a la estructura sugerida para tener una única fuente de verdad de los códigos y de los textos académicos, y un segundo juego de artefactos (`*_temprano`) para el modelo de alerta temprana.

## 6. Metodología

1. **Análisis exploratorio** (notebook): tipos de variables, estadísticas descriptivas, nulos y duplicados, histogramas y boxplots, distribución de la variable objetivo, correlaciones de Spearman, outliers (IQR), reglas de consistencia y variables asociadas con la deserción.
2. **Limpieza determinista** (`src/data_loader.py`): se eliminan el BOM y el tabulador de los nombres de columna, se renombran a `snake_case`, se validan tipos, rangos documentados (notas 0–20 y 0–200, etc.) y reglas lógicas (aprobadas ≤ inscritas). **No se eliminan valores extremos legítimos**: las edades altas corresponden a la vía de admisión «mayores de 23 años» y las notas 0 a semestres sin unidades aprobadas; eliminarlos sesgaría justo la clase Deserción.
3. **Partición** estratificada 80/20 con `random_state=42` (3.539 entrenamiento / 885 prueba).
4. **Pipelines sin fuga de información**: tasa de aprobación por semestre (transformación sin estado) + `ColumnTransformer` específico por algoritmo, ajustado dentro de cada pliegue:

   | Algoritmo | Nominales | Numéricas |
   |---|---|---|
   | Regresión Logística | One-Hot (categorías con < 10 casos → «infrecuente») | `StandardScaler` |
   | Random Forest | One-Hot | sin escalar |
   | HistGradientBoosting | Ordinal + categóricas nativas | sin escalar |

   Las categorías no vistas en entrenamiento se envían al grupo infrecuente (One-Hot) o a valor faltante (HGB): la predicción no falla.
5. **Desbalance de clases**: se incluye `class_weight` (sin ponderar vs. balanceado) en la rejilla de hiperparámetros y la validación cruzada decide. En los 6 casos (3 algoritmos × 2 modalidades) ganó la ponderación balanceada, lo que mejora el recall de la clase minoritaria *Matriculado*.
6. **Ajuste y selección**: `GridSearchCV` con `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, optimizando **macro-F1**. El ganador se elige por macro-F1 medio de validación cruzada; **el conjunto de prueba no interviene en la selección** y solo se usa para la evaluación final de los tres modelos.
7. **Alerta temprana**: se repite el protocolo completo con un segundo conjunto de columnas que **excluye las 6 variables del 2.º semestre**, generando su propio artefacto, metadatos y métricas.

## 7. Modelos

| Algoritmo | Rejilla evaluada |
|---|---|
| Regresión Logística (multinomial, lbfgs) | `C` ∈ {0,01; 0,1; 1; 10} × `class_weight` ∈ {None, balanced} |
| Random Forest (400 árboles) | `max_depth` ∈ {None, 15} × `min_samples_leaf` ∈ {1, 3} × `max_features` ∈ {sqrt, 0,3} × `class_weight` ∈ {None, balanced_subsample} |
| HistGradientBoostingClassifier (early stopping) | `learning_rate` ∈ {0,05; 0,1} × `max_leaf_nodes` ∈ {15, 31} × `l2_regularization` ∈ {0, 1} × `class_weight` ∈ {None, balanced} |

## 8. Métricas reales

Valores generados por `python -m src.train` (archivos `models/metricas.json` y `models/metricas_temprano.json`). CV = media ± desviación en 5 pliegues sobre entrenamiento; Prueba = 885 estudiantes nunca vistos.

### 8.1 Modelo con información completa (36 variables)

| Algoritmo | Macro-F1 (CV) | Accuracy (prueba) | Macro-F1 (prueba) | Balanced acc. (prueba) | Recall Deserción | Recall Matriculado | Recall Graduado | ROC-AUC OvR |
|---|---|---|---|---|---|---|---|---|
| Regresión Logística | 0,7169 ± 0,0143 | 0,7277 | 0,6877 | 0,6981 | 0,690 | 0,610 | 0,794 | 0,8849 |
| Random Forest | 0,7255 ± 0,0170 | 0,7672 | 0,7164 | 0,7157 | 0,729 | 0,547 | 0,871 | 0,8899 |
| **★ HistGradientBoosting** | **0,7282 ± 0,0158** | 0,7458 | 0,7054 | 0,7154 | 0,711 | 0,623 | 0,812 | 0,8803 |

**Modelo seleccionado: HistGradientBoostingClassifier** (`class_weight=balanced`, `learning_rate=0,05`, `max_leaf_nodes=15`, `l2_regularization=1`), por tener el mayor macro-F1 en validación cruzada.

Precisión / recall / F1 por clase en prueba del modelo seleccionado:

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Deserción | 0,815 | 0,711 | 0,759 | 284 |
| Matriculado | 0,442 | 0,623 | 0,517 | 159 |
| Graduado | 0,869 | 0,812 | 0,840 | 442 |

Matriz de confusión (filas = real, columnas = predicho; orden Deserción, Matriculado, Graduado): `[[202, 57, 25], [31, 99, 29], [15, 68, 359]]`.

> **Nota de interpretación.** En prueba, Random Forest obtuvo un macro-F1 ligeramente superior (0,7164 vs. 0,7054). No se cambió la elección: hacerlo sería seleccionar con el conjunto de prueba. Las diferencias en CV entre HGB y RF (0,003) son menores que la desviación entre pliegues, por lo que ambos modelos deben considerarse de desempeño similar. HGB ofrece además el mejor recall de la clase minoritaria *Matriculado*.

Variables de las que más depende el modelo (importancia por permutación en prueba, caída de macro-F1): unidades aprobadas del 2.º semestre (0,212), unidades aprobadas del 1.er semestre (0,067), matrícula al día (0,054), unidades inscritas del 2.º semestre (0,046) y programa académico (0,028).

### 8.2 Modelo de alerta temprana (cierre del 1.er semestre, 30 variables)

| Algoritmo | Macro-F1 (CV) | Accuracy (prueba) | Macro-F1 (prueba) | Balanced acc. (prueba) | Recall Deserción | Recall Matriculado | Recall Graduado |
|---|---|---|---|---|---|---|---|
| Regresión Logística | 0,6844 ± 0,0167 | 0,7096 | 0,6715 | 0,6813 | 0,673 | 0,597 | 0,774 |
| Random Forest | 0,6932 ± 0,0230 | 0,7333 | 0,6777 | 0,6786 | 0,697 | 0,497 | 0,842 |
| **★ HistGradientBoosting** | **0,6976 ± 0,0117** | 0,7198 | 0,6747 | 0,6810 | 0,715 | 0,541 | 0,787 |

Este modelo responde a otra pregunta (anticipar con información limitada) y no debe compararse directamente con el modelo completo.

## 9. Instalación

Requiere Python 3.11–3.13.

```bash
git clone https://github.com/DanielSQN/edupredict-ml-streamlit.git
cd edupredict-ml-streamlit
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # o requirements.txt si solo va a ejecutar la app
```

## 10. Entrenamiento

```bash
python -m src.train              # usa data/data.csv (≈3 min con 4 núcleos)
python -m src.train --download   # descarga el CSV de UCI si no existe
python -m src.train --n-jobs 2   # limitar procesos
```

Genera `data/estudiantes_limpio.csv` y los seis archivos de `models/`. El proceso es determinista: dos ejecuciones producen exactamente las mismas métricas.

Pruebas:

```bash
python -m pytest
```

## 11. Ejecución de Streamlit

```bash
streamlit run app.py
```

Abrir http://localhost:8501. La aplicación no entrena modelos: si faltan los artefactos, lo indica y sugiere ejecutar el entrenamiento.

## 12. Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub (incluye `models/` y `data/`).
2. En [share.streamlit.io](https://share.streamlit.io) → *Create app* → seleccionar el repositorio, la rama y el archivo principal `app.py`.
3. En *Advanced settings* elegir **Python 3.12 o 3.13**.
4. Desplegar. `requirements.txt` instala las dependencias; todas las rutas son relativas a la raíz del proyecto.

**Procedencia de los artefactos versionados.** Los archivos de `models/` se generaron con `python -m src.train` sobre `data/data.csv` (SHA-256 completo en `models/metadata.json`), con Python 3.13, scikit-learn 1.9.1, pandas 3.0.5 y numpy 2.5.3. `scikit-learn` está fijado a la misma versión en `requirements.txt` porque los archivos `joblib` no son portables entre versiones. Si cambia la versión, regenere los artefactos con `python -m src.train`.

### Póster

```bash
python scripts/build_poster.py   # regenera docs/poster/poster.html con las métricas de models/
```

Abrir `docs/poster/poster.html` en Chrome/Edge → *Imprimir → Guardar como PDF* (tamaño A1, sin márgenes, gráficos de fondo activados). Para completar el póster: agregar `docs/poster/logo_cun.png`, los códigos `docs/poster/qr_app.png`, `qr_video.png`, `qr_repositorio.png`, y llenar `INTEGRANTES`, `DOCENTE`, `ENLACE_APP` y `ENLACE_VIDEO` en `src/project_info.py`. El borrador actual (`docs/poster/poster_borrador.pdf`) tiene esos espacios marcados como pendientes.

## 13. Capturas de pantalla

| Dashboard | Predicción individual |
|---|---|
| ![Dashboard](docs/capturas/dashboard.png) | ![Predicción](docs/capturas/prediccion.png) |
| **Comparación de modelos** | **Acerca del proyecto** |
| ![Comparación](docs/capturas/comparacion.png) | ![Acerca](docs/capturas/acerca.png) |

## 14. Correcciones y decisiones metodológicas

- **Nombres de columna**: el CSV de UCI trae BOM y un tabulador en `Daytime/evening attendance\t`; se normalizan antes de cualquier uso.
- **Variables nominales codificadas como enteros** (programa, ocupaciones, etc.): se tratan como categorías (One-Hot / categóricas nativas), nunca como magnitudes.
- **Sin fuga de información**: escalado y codificación viven dentro del pipeline y se ajustan por pliegue.
- **Selección sin el conjunto de prueba**: el ganador se elige por CV; la prueba se usa una sola vez.
- **Desbalance**: tratado como hiperparámetro y evaluado con macro-F1 y balanced accuracy, no solo accuracy.
- **Outliers**: se conservan los valores extremos legítimos (justificación en la sección 6 y en el notebook).
- **Alerta temprana**: modelo independiente sin variables del 2.º semestre, con artefacto y métricas propios; la prueba `test_early_model_does_not_use_second_semester` lo verifica.
- **Coherencia notebook–pipeline–interfaz**: el notebook y la app importan los mismos módulos de `src/`; las pruebas verifican que `estudiantes_limpio.csv` sea idéntico a la salida de la limpieza y que la predicción desde la interfaz coincida con el pipeline directo.

## 15. Limitaciones

- Datos de una sola institución portuguesa (cohortes 2008/09–2018/19); los patrones pueden no trasladarse a otros contextos.
- La clase *Matriculado* es minoritaria y la más difícil (F1 ≈ 0,52).
- Las probabilidades no están calibradas para uso institucional; son estimaciones académicas, no una medición validada de riesgo.
- El modelo identifica asociaciones, no causas. Variables sensibles (sexo, nacionalidad) podrían introducir sesgos.

## 16. Referencias (APA 7)

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En *Trends and Applications in Information Systems and Technologies* (pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*, 3146–3154.

## 17. Enlaces públicos

- Repositorio: https://github.com/DanielSQN/edupredict-ml-streamlit
- Aplicación en Streamlit Community Cloud: *pendiente de despliegue*
- Video de sustentación: *pendiente*

## 18. Integrantes

*Pendiente: completar con los nombres completos de los integrantes (también en `src/project_info.py`).*
