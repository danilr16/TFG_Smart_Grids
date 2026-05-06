# ⚡ Smart Grids: Control de Redes Eléctricas con Inteligencia Artificial

Repositorio oficial del código fuente del Trabajo Fin de Grado (TFG) desarrollado por **Daniel López Ramos**. 

Este proyecto ha sido supervisado y guiado por los tutores **David Gutiérrez Avilés** y **Manuel Barragán Villarejo**, para la obtención del título de **Grado en Ingeniería Informática - Ingeniería del Software por la Universidad de Sevilla**.

Este proyecto investiga la aplicación de técnicas avanzadas de **Machine Learning** y **Deep Learning** para resolver el control activo de redes de distribución eléctrica (Smart Grids). El objetivo es reemplazar y/o asistir a los algoritmos matemáticos deterministas clásicos (OPF) con modelos orientados a datos capaces de aprender la causalidad física del sistema.

En este repositorio encontrarás el pipeline analítico completo: desde el procesamiento de señales eléctricas, hasta el diseño y evaluación de arquitecturas de IA complejas capaces de predecir la inyección óptima de potencia reactiva, las pérdidas de energía y las tensiones nodales de la red.

**Enlace a la memoria del proyecto:** [Smart Grids: Control de Redes Eléctricas con Inteligencia Artifical](/Control_Redes_Eléctricas_Con_IA.pdf)

---

## 🎯 Objetivo del Proyecto

El sistema busca optimizar la operación de la red seleccionada prediciendo con alta precisión:
1.  **Potencia Reactiva Inyectada en el Sistema**
2.  **Pérdidas del Sistema**
3.  **Tensiones Nodales**

Para lograrlo, se exploran modelos clásicos (Random Forest, XGBoost), redes neuronales profundas (Keras/TensorFlow) y se propone una solución híbrida de desarrollo propio: la arquitectura **Electric Sequential Regressor (ESR)**.

---

## 🏗️ Arquitectura y Modelos

El flujo de trabajo se estructura en forma de módulos independientes ejecutados sobre cuadernos de Jupyter (`.ipynb`). Se evalúan las siguientes arquitecturas predictivas:

*   🌲 **Modelos de Ensamblaje Estáticos:** `RandomForest` y `XGBoost`.
*   🧠 **Red Neuronal Multisalida (Deep Learning):** Topología diseñada con la API funcional de Keras que predice simultáneamente las tres variables objetivo utilizando pesos asimétricos (2.0 para las pérdidas y 1.0 para el resto) y una estandarización adaptativa.
*   🔄 **Electric Sequential Regressor (ESR):** Arquitectura principal del proyecto desarrollada a medida. Se trata de un modelo jerárquico compatible con Scikit-Learn que emula el flujo de cargas físico:
    *   **Fase 1 (Pivote):** Un primer estimador deduce la inyección de reactiva.
    *   **Fase 2 (Propagación):** Un segundo estimador concatena la entrada original con la predicción de la Fase 1 para estimar las variables de estado complejas (tensiones y pérdidas).

### Validación Robusta y Anti Data Leakage
Dada la naturaleza temporal de los datos eléctricos, el pipeline *no usa k-folds aleatorios*. Todos los modelos se validan y optimizan utilizando `TimeSeriesSplit` (Validación Cruzada Temporal) sobre un conjunto de retención cronológico inicial del 80% (aislando el 20% más reciente de los datos como conjunto de pruebas ciego definitivo).

---

## 💻 Entorno y Tecnologías Utilizadas

Este proyecto utiliza el ecosistema estándar de Data Science en Python: 

*   **Lenguaje:** Python 3.11+
*   **Gestión de Datos y Álgebra:** Pandas, NumPy
*   **Visualización:** Matplotlib, Seaborn
*   **Machine Learning Base:** Scikit-Learn
*   **Deep Learning:** TensorFlow, Keras
*   **Gradient Boosting:** XGBoost
*   **Optimización de Hiperparámetros:** Keras Tuner, MetaGen

---

## 🛠️ Instalación y Configuración

Se recomienda desplegar este entorno de forma aislada.

**1. Clonar el repositorio y crear el entorno virtual:**
```bash
git clone https://github.com/danilr16/TFG_Smart_Set_Points.git
cd entrenamiento
python -m venv .venv
```

**2. Activar el entorno virtual:**

    Windows (PowerShell): .\.venv\Scripts\Activate.ps1

    Linux/macOS: source .venv/bin/activate

**3. Instalar las dependencias externas:**
```bash
pip install jupyter
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tensorflow keras keras-tuner xgboost
pip install pymetagen-datalabupo
```

`(Nota: Extraído del Manual de Instalación Anexo A del TFG)`

**4. Instalar el código base (src) como paquete editable:**

Para que los cuadernos interactivos puedan acceder a las clases desarrolladas (como el ESR), ejecuta desde la raíz:
```bash
pip install -e .
```

`(Asegúrate de estar en el mismo directorio donde se ubica setup.py)`

**5. Lanzar el entorno:**

```bash
jupyter notebook
```

## 📂 Estructura del Proyecto

    /Memoria: Código de la memoria en LaTeX con todos los recursos utilizados por esta.
        /figures: Imágenes utilizadas en la memoria.
        /tables: Tablas utilizadas en la memoria.
        /sections: Secciones de la memoria.
        bibliografia.bib: Bibliografía utilizada para la memoria.
        TFG.tex: Plantilla base que define la estructura de la memoria en LaTeX.
    /Proyecto:
        /comprimidos: Comprimidos originales distribuido por los tutores para la generación del dataset. (Contenido no desarrollado por el alumno).
        /creacion_dataset: Código distribuido por los tutores para la generación del dataset. (Contenido no desarrollado por el alumno).
        /entrenamiento: Código fuente del proyecto. Contiene los diferentes experimentos y el código base desarrollado. (Contenido aportado por el alumno).