# 🎙️ IA Audio Denoising: Cancelación de Ruido con Deep Learning

Este proyecto implementa una arquitectura de Deep Learning basada en un **Denoising Autoencoder Convolucional** para identificar y eliminar el ruido de fondo estático en grabaciones de voz humana. 

El sistema transforma el problema acústico en un problema de visión artificial convirtiendo los archivos de audio en espectrogramas mediante la Transformada de Fourier a Corto Plazo (STFT). Tras el filtrado espacial, el audio se reconstruye utilizando la Transformada Inversa (ISTFT).

## 🚀 Características Principales
* **Preprocesamiento Automático:** Conversión de audios `.wav` a espectrogramas normalizados.
* **Descarga Dinámica:** Integración con `kagglehub` para descargar el dataset masivo directamente en el entorno de ejecución sin saturar el almacenamiento local de GitHub.
* **Interfaz Web interactiva:** Aplicación desarrollada con Gradio para procesar audios en tiempo real desde el micrófono o mediante archivos subidos.

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto en tu máquina local, asegúrate de tener Python 3.8+ instalado. 

1. Clona este repositorio:
`git clone https://github.com/Negerty48/voice-denoising-autoencoder`
`cd voice-denoising-autoencoder`

1. Instala las dependencias necesarias. Se recomienda crear un entorno virtual previamente:
`pip install -r requirements.txt`

## 📂 Estructura del Proyecto

* `notebook_entrenamiento.ipynb`: Jupyter Notebook que contiene todo el proceso de exploración de datos, descarga del dataset, preprocesamiento de la STFT, definición del Autoencoder, entrenamiento (con *Early Stopping*) y evaluación de métricas (MSE/MAE).
* `app.py`: Script principal que lanza la interfaz web interactiva para probar el modelo en directo.
* `data/`: Carpeta (ignorada en git) donde se guardarán los audios y los espectrogramas generados.
* `models/`: Carpeta donde se guardan los modelos creados en formato keras.

## ⚙️ Cómo ejecutar el proyecto

### Fase 1: Entrenamiento del Modelo
1. Abre el archivo `notebook_entrenamiento.ipynb` en Jupyter, VSCode o Google Colab.
2. Ejecuta las celdas en orden. El script de `kagglehub` descargará automáticamente el dataset de 5GB.
3. Al finalizar, el modelo entrenado se guardará automáticamente en `models/denoising_autoencoder.keras`.

### Fase 2: Aplicación Web en Directo
Una vez entrenado el modelo, puedes lanzar la interfaz gráfica:
1. Abre una terminal en la raíz del proyecto.
2. Ejecuta el siguiente comando: `python app.py`
3. Se generará un enlace local (usualmente `http://127.0.0.1:7860`). Ábrelo en tu navegador web.
4. Sube un archivo de audio ruidoso o graba tu voz con el micrófono, presiona "Limpiar Audio con IA" y escucha el resultado.

## ⚠️ Limitaciones y Mejoras Futuras
Debido a la naturaleza de la reconstrucción mediante ISTFT, el modelo actual recicla la *fase* del audio ruidoso original para combinarse con la *magnitud* limpia predicha por la red. Esto puede generar un ligero artefacto acústico (tono robótico).

Como trabajo futuro, se propone migrar el preprocesamiento a Espectrogramas de Mel e integrar un **Vocoder Neuronal (ej. HiFi-GAN)** para sintetizar la fase acústica desde cero, devolviendo una calidez y naturalidad 100% humana a la señal final.