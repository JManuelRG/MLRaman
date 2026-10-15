# “Desarrollo e implementación de un modelo de inteligencia artificial para el análisis de ciprofloxacino mediante espectroscopía Raman” 

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-notebooks-F9AB00?logo=googlecolab&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)

**Instituto Politécnico Nacional · Unidad Profesional Interdisciplinaria de Ingeniería Campus Guanajuato (UPIIG)**
Ingeniería Farmacéutica 

Alumno: 
Martinez Balderas Cesar Israel 

Asesores: 
José Manuel Garnica Rivera 
Juan Carlos Martinez Espinosa 

---

## Resumen

El ciprofloxacino es una fluoroquinolona de amplio espectro de uso muy extendido. Su control de calidad se realiza habitualmente con técnicas cromatográficas (HPLC, TLC), que son precisas pero lentas, costosas y destructivas.

La **espectroscopía Raman** ofrece una alternativa rápida, no destructiva y sin preparación especial de la muestra, aunque la interpretación de sus espectros se complica por la fluorescencia, los rayos cósmicos (*spikes*) y el solapamiento de bandas.

Este proyecto desarrolla un **flujo reproducible de preprocesamiento, quimiometría y aprendizaje automático** que, a partir de espectros Raman de ciprofloxacino, estima la concentración del fármaco y apoya su identificación y control de calidad. Se comparan tres enfoques: **PLS** (referencia lineal), **Random Forest** y una **red neuronal profunda**.

> **Hipótesis.** El uso de un modelo de inteligencia artificial permite analizar los espectros Raman de ciprofloxacino de manera precisa y reducir el tiempo de análisis para la identificación y el control de calidad en la industria farmacéutica.

## Objetivos

**Objetivo general.** Desarrollar e implementar una herramienta basada en inteligencia artificial para el análisis de espectros Raman de ciprofloxacino, orientada a la identificación, el control de calidad y la evaluación de muestras problema.

**Objetivos específicos**

- Construir una base de datos espectral de ciprofloxacino que considere distintos laboratorios, concentraciones y condiciones de adquisición.
- Desarrollar modelos de IA en Python (scikit-learn, TensorFlow/Keras) para el análisis de los espectros.
- Evaluar el desempeño de los modelos con métricas estadísticas y validar su aplicabilidad en el control de calidad farmacéutico.

## Datos

Se reutilizan los espectros de la investigación previa de **Martínez-Espinosa et al. (2019)**.

| Característica | Valor |
|---|---|
| Número de espectros | 600 |
| Variables de intensidad (por espectro) | 3319 |
| Rango espectral | 99.97 – 3299.30 cm⁻¹ |
| Estructura | 8 bloques (laboratorios A–H) de 75 espectros cada uno |
| Controles | 0, 20, 40, 60, 80 y 100 % |
| Excipiente | Almidón |
| Fármaco de referencia | Ciprofloxacino puro (Sigma-Aldrich, 98 %) |

Los datos se organizan en una matriz espectral **Ψ ∈ ℝ^(3319×600)**, donde cada columna es un espectro y cada fila un número de onda.

> Los datos originales no se incluyen en este repositorio sin autorización de los autores de la investigación previa. Consulta [Licencia y uso de datos](#licencia-y-uso-de-datos).

### Bandas Raman de interés

| Banda (cm⁻¹) | Asignación | Grupo funcional |
|---|---|---|
| 634 | Deformación C–H | Núcleo quinolónico |
| 749 | Balanceo de CH₂ | Estructura molecular |
| 1000–1030 | Respiración de anillo | Ciclopropilo |
| 1000–1100 | Estiramiento C–F | Sustituyente fluorado |
| 1268 | Respiración de anillo y flexión C–H | Piperazina |
| 1384 | Balanceo de CH₂ | Piperazina |
| 1621 | Estiramiento C=C | Núcleo quinolónico |

Fuente: elaboración propia a partir de Liu et al. (2022).

## Metodología

```
Datos crudos  ──►  Matriz Ψ  ──►  Preprocesamiento  ──►  División  ──►  PCA  ──►  Modelos  ──►  Evaluación  ──►  Muestra problema
(.txt)            (3319×600)     spikes · suavizado     70/15/15      reducción   PLS · RF · DNN   RMSE · R² · MAE   predicción de
                                 · línea base ALS       estratificada  y outliers                    + interpretabilidad  concentración
```

### 1. Preprocesamiento

1. **Eliminación de picos cósmicos.** Algoritmo adaptado de Whitaker y Hayes (2018): diferencias consecutivas, ruido local con desviación absoluta mediana (MAD, ventana de 101 puntos, umbral 8) y una **máscara de protección** para las bandas características del fármaco (puntos marcados en más del 85 % de los espectros).
2. **Suavizado** mediante filtro de mediana.
3. **Corrección de línea base** con mínimos cuadrados asimétricos (ALS), con λ = 10⁶ seleccionado tras evaluar el rango 10⁴–10⁷.
4. **Normalización** y escalado con `StandardScaler` ajustado únicamente con los datos de entrenamiento.

### 2. División de datos

Entrenamiento 70 %, validación 15 % y prueba 15 %, de forma estratificada. El conjunto de prueba permanece aislado hasta la evaluación final.

### 3. Análisis exploratorio (PCA)

Reducción de dimensionalidad (varianza acumulada ≥ 95 % o *scree plot*), detección de valores atípicos y visualización de agrupamientos.

### 4. Modelos predictivos

| Modelo | Descripción | Ajuste de hiperparámetros |
|---|---|---|
| **PLS** | Referencia lineal | Validación cruzada de 5 pliegues (RMSECV) |
| **Random Forest** | No lineal, basado en árboles | Búsqueda en rejilla con validación cruzada |
| **Red neuronal profunda** | Totalmente conectada: 128-64-32, ReLU, *dropout* 0.3 y 0.2, salida lineal | Adam, pérdida MSE, *early stopping* (máx. 200 épocas) |

### 5. Evaluación e interpretabilidad

- Métricas: **RMSE, R², MAE**, además de análisis de residuos.
- Interpretación: *loadings* (PLS), importancia de características (Random Forest) y **SHAP** (red neuronal).
- Intervalos de confianza por *bootstrap* sobre el conjunto de prueba.
- Reproducibilidad: semilla aleatoria fija y notebook documentado.

## Estado del proyecto

- [x] Fase I. Recopilación y construcción de la matriz espectral
- [x] Fase II. Preprocesamiento: eliminación de *spikes* y corrección de línea base
- [ ] Normalización y suavizado final
- [ ] Fase III. Reducción de dimensionalidad (PCA)
- [ ] Fase IV. Modelado predictivo comparativo (PLS, Random Forest, DNN)
- [ ] Evaluación, selección del modelo y aplicación a muestra problema

## Estructura del repositorio

> Estructura propuesta; se irá completando conforme avance el proyecto.

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # datos originales (no versionados)
│   └── processed/        # matrices procesadas
├── notebooks/
│   ├── 01_construccion_matriz.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_pca.ipynb
│   └── 04_modelado.ipynb
├── src/                  # funciones reutilizables (spikes, ALS, métricas)
├── figures/              # gráficas exportadas
├── docs/                 # anteproyecto y reportes
└── models/               # modelos entrenados
```

## Instalación y uso

### Opción A: Google Colab

Abre el notebook deseado desde la carpeta `notebooks/` con el botón *Open in Colab* de GitHub, o súbelo manualmente a Colab.

### Opción B: entorno local

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

Dependencias principales: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `shap`, `jupyter`.

## Resultados preliminares

- Matriz espectral consolidada y verificada: 600 espectros × 3319 variables.
- Un filtro de mediana con umbral estático y una versión con doble ventana (7 y 41 puntos, k = 8) resultaron inadecuados: sobrefiltraban regiones ruidosas.
- El algoritmo adaptado de Whitaker y Hayes, junto con la máscara de protección, detecta *spikes* reales sin alterar las bandas de 635 y 1623 cm⁻¹ del ciprofloxacino.
- La corrección ALS con λ = 10⁶ equilibra el seguimiento de la fluorescencia y la preservación de las bandas.

## Cronograma

| Etapa | Periodo | Actividades principales |
|---|---|---|
| Proyecto Terminal I | Ene–Jun | Definición del tema, anteproyecto, marco teórico y metodología |
| Proyecto Terminal II | Ago–Dic | Preprocesamiento, modelado ML/DL, evaluación y muestra problema |
| Proyecto Terminal III | Ene–May | Mejoras al código, validación continua, comparación con la bibliografía y conclusiones |

## Limitaciones

- Los notebooks dependen de Google Colab y de conexión a internet; pueden ejecutarse localmente con Jupyter, Spyder o VS Code.
- El equipo Raman no está disponible en la UPIIG, lo que condiciona la adquisición de nuevos espectros.
- La recolección de muestras adicionales requiere tiempo.

## Equipo

| Rol | Nombre |
|---|---|
| Alumno | César Israel Martínez Balderas |
| Asesor | José Manuel Garnica Rivera |
| Asesor | Juan Carlos Martínez Espinosa |
| Revisora | Ana Lidia Quintero Ramírez |

## Referencias principales

- Barton, B. et al. (2022). *Chemometrics for Raman Spectroscopy Harmonization*. Applied Spectroscopy, 76(9), 1021–1041.
- Liu, C. et al. (2022). *Raman-based detection of ciprofloxacin and its degradation in pharmaceutical formulations*. Talanta, 250, 123719.
- Martínez-Espinosa, J. C. et al. (2019). *Generation of Raman images through spectral mappings*. Journal of Intelligent & Fuzzy Systems, 36(5), 4991–4999.
- Whitaker, D. A. y Hayes, K. (2018). Algoritmo de detección de rayos cósmicos en espectros Raman.
- Vulchi, R. T. et al. (2024). *Artifacts and anomalies in Raman spectroscopy*. Molecules, 29(19), 4748.
- Pence, I. y Mahadevan-Jansen, A. (2016). *Clinical instrumentation and applications of Raman spectroscopy*. Chemical Society Reviews, 45(7), 1958–1979.

La bibliografía completa se encuentra en el anteproyecto (`docs/`).

## Licencia y uso de datos

- **Código:** MIT
- **Datos:** provienen de una investigación previa (Martínez-Espinosa et al., 2019). No deben redistribuirse sin el consentimiento de sus autores. Si el repositorio es público, los datos se omiten con `.gitignore` y se documenta cómo obtenerlos.

## Cómo citar

Si utilizas este trabajo, cítalo como:

> Martínez Balderas, C. I. (2026). *Desarrollo e implementación de un modelo de inteligencia artificial para el análisis de ciprofloxacino mediante espectroscopía Raman* [Proyecto Terminal]. UPIIG, Instituto Politécnico Nacional.
