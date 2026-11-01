# Contenido base del póster académico — EduPredict AI

> Borrador visual: `docs/poster/poster.html` → `docs/poster/poster_borrador.pdf` (A1 vertical).
> Los elementos entre corchetes **[ ]** requieren información de los integrantes. No se inventaron nombres, logos ni enlaces.

---

## Encabezado

- **[LOGO OFICIAL CUN]** (insertar archivo oficial en `docs/poster/logo_cun.png`)
- **Título:** EduPredict AI — Predicción de deserción universitaria con Machine Learning clásico
- **Integrantes:** [Nombre completo 1], [Nombre completo 2], [Nombre completo 3]
- **Docente:** [Nombre del docente]
- Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN · [Periodo académico]

## Problema

La deserción universitaria suele detectarse cuando ya es tarde para intervenir.
**¿Cómo transformar datos educativos en conocimiento predictivo para anticipar la deserción de un estudiante y facilitar la identificación de posibles situaciones de riesgo académico?**

## Objetivo

Desarrollar una solución de Machine Learning clásico que permita analizar información académica, entrenar y comparar modelos de clasificación, evaluar su desempeño y presentar predicciones mediante una aplicación interactiva de Streamlit.

## Dataset

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
| Regresión Logística | 0,717 | 0,728 | 0,688 | 0,698 |
| Random Forest | 0,726 | 0,767 | 0,716 | 0,716 |
| **★ HistGradientBoosting (seleccionado)** | **0,728** | 0,746 | 0,705 | 0,715 |

Recall del modelo seleccionado: Deserción 0,711 · Matriculado 0,623 · Graduado 0,812.

Alerta temprana (sin 2.º semestre, HistGradientBoosting): macro-F1 CV 0,698 · accuracy 0,720 · macro-F1 0,675 · balanced acc. 0,681.

## Captura de Streamlit

`docs/capturas/dashboard.png`, `docs/capturas/prediccion.png`, `docs/capturas/comparacion.png` (capturas reales de la aplicación).

## Conclusiones

1. Los tres algoritmos alcanzan desempeños cercanos (macro-F1 CV 0,717–0,728); HistGradientBoosting fue seleccionado por validación cruzada, con diferencia menor a la variabilidad entre pliegues frente a Random Forest.
2. Las unidades aprobadas por semestre y tener la matrícula al día son las variables de las que más depende el modelo.
3. *Matriculado* es la clase más difícil (F1 0,52): su perfil es intermedio entre graduados y desertores.
4. Con solo la información del 1.er semestre es posible una alerta temprana con macro-F1 de 0,675 en prueba, un semestre antes.
5. La herramienta es académica: identifica asociaciones, no causas, y no reemplaza decisiones institucionales.

## Referencias (APA 7)

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

## Enlaces y códigos QR

- Aplicación Streamlit: [URL pública] → [QR]
- Video: [URL pública] → [QR]
- Repositorio: https://github.com/DanielSQN/edupredict-ml-streamlit → [QR]
