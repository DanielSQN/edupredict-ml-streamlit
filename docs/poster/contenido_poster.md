# Contenido base del póster académico — EduPredict AI

> Borrador visual: `docs/poster/poster.html` → `docs/poster/poster_borrador.pdf` (A1 vertical).
> Los elementos entre corchetes **[ ]** siguen pendientes (logo, QR, enlaces públicos y periodo académico).

---

## Encabezado

- **[LOGO OFICIAL CUN]** (insertar archivo oficial en `docs/poster/logo_cun.png`)
- **Título:** EduPredict AI — Predicción de deserción universitaria con Machine Learning clásico
- **Integrantes:** Paula Carolina Calvo Pérez, Juan Sebastián Caviedes Roa, Jonathan Camilo Flórez Bolívar, Daniel Santiago Quintero Niño
- **Docente:** Dr. Hamilton Rivera Flor
- Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN · [Periodo académico]

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

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

## Enlaces y códigos QR

- Aplicación Streamlit: [URL pública] → [QR]
- Video: [URL pública] → [QR]
- Repositorio: https://github.com/DanielSQN/edupredict-ml-streamlit → [QR]
