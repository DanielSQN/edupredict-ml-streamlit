# Guion del video de sustentación (3 a 5 minutos)

> Estructura exigida por el ACA: problema, preparación de datos, métricas y demostración en vivo de Streamlit.
> Las cifras provienen de `models/metricas.json` y `models/metricas_temprano.json`. Ajustar quién presenta cada bloque.

| Tiempo | Bloque | Qué decir / mostrar |
|---|---|---|
| 0:00–0:30 | **Presentación y problema** | Integrantes [nombres], pilar Educación. La deserción suele detectarse tarde. Pregunta orientadora: ¿cómo transformar datos educativos en conocimiento predictivo para anticipar la deserción? |
| 0:30–1:10 | **Dataset y EDA** | UCI 697: 4.424 estudiantes, 36 variables, 3 clases (Deserción 32,1 %, Matriculado 17,9 %, Graduado 49,9 %). Mostrar el Dashboard: matrícula al día (86,6 % vs. 24,7 % de deserción), beca (38,7 % vs. 12,2 %) y rendimiento semestral como señales principales. |
| 1:10–1:50 | **Preparación de datos** | 0 nulos y 0 duplicados; nombres normalizados (tab oculto en una columna); reglas de consistencia; outliers legítimos conservados (la winsorización se evaluó por validación cruzada). Partición estratificada 80/20 y pipelines sin fuga de información. |
| 1:50–2:40 | **Modelos y métricas** | Regresión Logística, Random Forest y Gradient Boosting con GridSearchCV de 5 pliegues optimizando macro-F1. Modelo seleccionado por CV: **Gradient Boosting (HistGradientBoosting)** (macro-F1 CV 0,730). En prueba: accuracy 0,729, macro-F1 0,688, balanced accuracy 0,697. Mostrar la sección Comparación de modelos y la matriz de confusión: Matriculado es la clase más difícil. |
| 2:40–3:10 | **Alerta temprana** | Modelo independiente sin datos del 2.º semestre: macro-F1 0,669 en prueba. Permite anticipar un semestre antes con poca pérdida. |
| 3:10–4:20 | **Demostración en vivo** | Sección Predicción individual: elegir modalidad, completar el formulario, pulsar *Analizar estudiante*, explicar la clase estimada, las probabilidades y la advertencia de herramienta académica. Cambiar «Matrícula al día» o las unidades aprobadas y mostrar cómo cambia la estimación. |
| 4:20–4:50 | **Conclusiones y limitaciones** | Los tres modelos rinden parecido; el rendimiento académico y la situación financiera son lo más asociado. Datos de una sola institución portuguesa; asociación no es causalidad; uso solo como apoyo. |
| 4:50–5:00 | **Cierre** | Enlaces a la app, al repositorio y al póster. |

**Consejos:** grabar la pantalla a 1080p con la app ya abierta (en Streamlit Community Cloud o con `streamlit run app.py`), y ensayar para no superar los 5 minutos.
