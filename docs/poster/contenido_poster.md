# Contenido base del póster académico — EduPredict AI

> Póster final: `docs/poster/poster.html` → `docs/poster/poster_EduPredictAI.pdf` (A1 vertical).
> El periodo académico no aparece en el póster final; agréguelo si el docente lo exige.

---

## Encabezado

- Logo oficial CUN: `docs/poster/logo_cun.png`
- **Título:** EduPredict AI — Predicción de deserción universitaria con Machine Learning clásico
- **Integrantes:** Paula Carolina Calvo Pérez, Juan Sebastián Caviedes Roa, Jonathan Camilo Flórez Bolívar, Daniel Santiago Quintero Niño
- **Docente:** Dr. Hamilton Rivera Flor
- Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN

## Problema

La deserción universitaria suele detectarse cuando ya es tarde para intervenir.
**¿Cómo transformar datos educativos en conocimiento predictivo para anticipar la deserción de un estudiante y facilitar la identificación de posibles situaciones de riesgo académico?**

## Objetivo

Desarrollar una solución de Machine Learning clásico que permita analizar información académica, entrenar y comparar modelos de clasificación, evaluar su desempeño y presentar predicciones mediante una aplicación interactiva de Streamlit.

## Pilar y dataset

- **Pilar:** Proyecto 2 — Educación (analítica académica y prevención de la deserción).

- *Predict Students' Dropout and Academic Success* — UCI ML Repository (ID 697).
- 4.424 estudiantes · 36 variables (académicas, socioeconómicas, demográficas, macroeconómicas) · sin nulos ni duplicados.
- Objetivo de 3 clases: Deserción 32,1 % · Matriculado 17,9 % · Graduado 49,9 %.

## Diagrama metodológico

```
Datos UCI → Limpieza determinista → Split estratificado 80/20 → Pipelines (sin fuga)
→ GridSearchCV 5-fold (macro-F1) → Selección por CV → Evaluación en prueba → Streamlit
```

## Algoritmos

1. Regresión Logística (One-Hot + estandarización).
2. Random Forest (One-Hot).
3. HistGradientBoosting (categóricas nativas).
Desbalance: `class_weight` como hiperparámetro (en todos los casos ganó la ponderación balanceada).

## Métricas reales (conjunto de prueba, n = 885)

| Modelo (información completa) | Macro-F1 CV | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---|---|---|
| **★ Gradient Boosting (HistGradientBoosting) (seleccionado)** | 0,730 | 0,729 | 0,688 | 0,697 |
| Random Forest | 0,724 | 0,769 | 0,717 | 0,716 |
| Regresión Logística | 0,717 | 0,724 | 0,684 | 0,695 |

Recall del modelo seleccionado: Deserción 0,697 · Matriculado 0,597 · Graduado 0,796.

Alerta temprana (sin 2.º semestre, Gradient Boosting): macro-F1 CV 0,699 · accuracy 0,711 · macro-F1 0,669 · balanced acc. 0,677.

## Captura de Streamlit

`docs/capturas/dashboard.png`, `docs/capturas/prediccion.png`, `docs/capturas/comparacion.png` (capturas reales de la aplicación).

## Conclusiones

1. Los tres algoritmos alcanzan desempeños cercanos (macro-F1 CV 0,717–0,730); Gradient Boosting fue seleccionado por validación cruzada, con diferencia menor a la variabilidad entre pliegues.
2. Las unidades aprobadas por semestre y tener la matrícula al día son las variables de las que más depende el modelo.
3. *Matriculado* es la clase más difícil (F1 0,49): su perfil es intermedio entre graduados y desertores.
4. Con solo la información del 1.er semestre es posible una alerta temprana con macro-F1 de 0,669 en prueba, un semestre antes.
5. La herramienta es académica: identifica asociaciones, no causas, y no reemplaza decisiones institucionales.

## Referencias (APA 7)

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*, 3146–3154.
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En Á. Rocha, H. Adeli, G. Dzemyda, F. Moreira, & A. M. Ramalho Correia (Eds.), *Trends and applications in information systems and technologies* (pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), Artículo 146. https://doi.org/10.3390/data7110146
- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Streamlit Inc. (2026). *Streamlit* (Versión 1.66.1) [Software]. https://streamlit.io

## Enlaces y códigos QR

- Aplicación Streamlit: https://edupredict-ml-app-cv2fdreafkspczy39oa7nu.streamlit.app/ → `qr_app.png`
- Video: https://www.youtube.com/watch?v=BEmN8dqMv-o → `qr_video.png`
- Repositorio: https://github.com/DanielSQN/edupredict-ml-streamlit → `qr_repositorio.png`
