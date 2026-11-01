"""Textos académicos del proyecto (sección "Acerca del proyecto" y póster).

Completar ``INTEGRANTES``, ``DOCENTE`` y los enlaces públicos cuando estén
disponibles. No se rellenan con datos inventados.
"""

NOMBRE = "EduPredict AI"
DESCRIPCION = (
    "Sistema inteligente de análisis académico y predicción de deserción universitaria "
    "mediante algoritmos clásicos de Machine Learning."
)

CONTEXTO = (
    "Proyecto ACA N.º 1 de la asignatura Aprendizaje Automático — Modelado con algoritmos "
    "clásicos de Machine Learning y despliegue interactivo en Streamlit. Especialización en "
    "Inteligencia Artificial, Escuela de Ingeniería, Corporación Unificada Nacional de "
    "Educación Superior (CUN)."
)

PILAR = "Proyecto 2 — Educación: analítica académica y prevención de la deserción"

PREGUNTA = (
    "¿Cómo transformar datos educativos en conocimiento predictivo para anticipar la "
    "deserción de un estudiante y facilitar la identificación de posibles situaciones "
    "de riesgo académico?"
)

OBJETIVO = (
    "Desarrollar una solución de Machine Learning clásico que permita analizar información "
    "académica, entrenar y comparar modelos de clasificación, evaluar su desempeño y "
    "presentar predicciones mediante una aplicación interactiva de Streamlit."
)

# Completar con los nombres completos de los integrantes, p. ej. ["Nombre Apellido", ...]
INTEGRANTES: list[str] = []
DOCENTE: str = "Dr. Hamilton Rivera Flor"

ENLACE_APP: str = ""     # URL pública en Streamlit Community Cloud
ENLACE_VIDEO: str = ""   # URL pública del video de sustentación
ENLACE_REPO: str = "https://github.com/DanielSQN/edupredict-ml-streamlit"

# Ruta de aprendizaje del ACA (enunciado oficial) y su implementación en este proyecto
RUTA_ACA = [
    ("Paso 1 · Exploración y EDA", "Notebook, Parte 1"),
    ("Paso 2 · Limpieza y datos", "Notebook, Parte 2 · src/data_loader.py"),
    ("Paso 3 · Modelado clásico", "Notebook, Partes 3–4 · src/train.py"),
    ("Paso 4 · Evaluación y métricas", "Notebook, Parte 4 · src/evaluate.py"),
    ("Paso 5 · Despliegue Streamlit", "app.py"),
]

METODOLOGIA = [
    ("Comprensión de los datos", "Exploración del dataset UCI 697: tipos de variables, estadísticas, distribución de clases, correlaciones y valores extremos."),
    ("Limpieza determinista", "Normalización de nombres de columnas, tipos y verificación de nulos, duplicados, rangos y reglas lógicas. No se eliminan extremos legítimos."),
    ("Partición", "División estratificada 80/20 (semilla 42). El 20 % de prueba queda aislado hasta la evaluación final."),
    ("Pipelines", "Ingeniería de variables sin estado + ColumnTransformer específico por algoritmo, ajustado dentro de cada pliegue para evitar fuga de información."),
    ("Ajuste y selección", "GridSearchCV con validación cruzada estratificada de 5 pliegues optimizando macro-F1; el desbalance se trata como hiperparámetro (pesos balanceados o no)."),
    ("Evaluación", "Accuracy, balanced accuracy, macro-F1, precisión/recall/F1 por clase y matriz de confusión en el conjunto de prueba."),
    ("Despliegue", "Pipeline ganador serializado con joblib y servido en Streamlit sin reentrenar."),
]

LIMITACIONES = [
    "Los datos provienen de una sola institución de educación superior de Portugal (cohortes 2008/09–2018/19); los patrones pueden no trasladarse a otros contextos, como el colombiano.",
    "La clase «Matriculado» (estudiantes que siguen inscritos al final de la duración normal del programa) es la minoritaria y la más difícil de distinguir.",
    "Las probabilidades del modelo no están calibradas para uso institucional ni constituyen una medición validada de riesgo; son estimaciones estadísticas con fines académicos.",
    "El modelo identifica asociaciones, no causas. La importancia de una variable no implica que modificarla cambie el resultado del estudiante.",
    "Las variables sensibles (sexo, nacionalidad, estado civil) pueden introducir sesgos; cualquier uso real requeriría un análisis de equidad.",
    "Los indicadores macroeconómicos corresponden al año de ingreso y solo toman 10 combinaciones distintas en el dataset.",
]

REFERENCIAS_APA = [
    "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89",
    "Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. Data, 7(11), 146. https://doi.org/10.3390/data7110146",
    "Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En Trends and Applications in Information Systems and Technologies (pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.",
    "Streamlit Inc. (2026). Streamlit [Software]. https://streamlit.io",
]
