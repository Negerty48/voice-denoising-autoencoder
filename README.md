# Speech Enhancement en el Dominio Complejo mediante Autoencoder Bi-Canal

Este proyecto implementa un sistema avanzado de supresión de ruido y mejora de voz (*Speech Enhancement*) basado en Aprendizaje Profundo. A diferencia de las aproximaciones tradicionales que solo procesan la magnitud espectral, este modelo opera en el plano complejo, permitiendo una reconstrucción lineal y estable tanto de la amplitud como de la fase acústica.

---

## 🚀 Evolución del Proyecto y Arquitectura

El desarrollo de este sistema ha seguido un proceso iterativo de investigación y resolución de cuellos de botella físicos y matemáticos propios del procesamiento de señales de audio con redes neuronales:

### 1. Autoencoder de Magnitud Conv2D (Línea Base)
* **Aproximación:** Transformación de la onda de audio 1D mediante STFT (*Short-Time Fourier Transform*) para extraer la matriz de **Magnitud** (tratada como una imagen 2D) y la **Fase** (apartada temporalmente).
* **Problema encontrado (Difuminado MSE):** Al entrenar con el Error Cuadrático Medio (*MSE*), ante la duda entre varias frecuencias, la red tiende a promediar de forma aritmética para minimizar el riesgo matemático. Esto genera espectrogramas borrosos que acústicamente se traducen en una voz apagada ("debajo del agua").

### 2. cGAN + Métodos de Reconstrucción Externa
Para solucionar el difuminado visual, se implementó una **GAN Condicional (cGAN)** con una función de pérdida adversarial (un Discriminador que actúa como juez). Aunque logró una nitidez visual perfecta en la magnitud, la reconstrucción del audio falló por los límites de la fase:
* **Fase Original Sucia:** Combinar la magnitud limpia de la cGAN con la fase ruidosa original provocaba un molesto "ruido de burbujas".
* **Algoritmo Griffin-Lim:** Al ser una aproximación heurística basada en proyecciones matemáticas y carecer de intuición acústica, dejaba un residuo metálico (*Phasiness*) similar a hablar por un tubo.
* **Neural Vocoders (HiFi-GAN):** Los vocoders neuronales se entrenan con voces perfectas de estudio. Al recibir el espectrograma con las micro-imperfecciones de nuestra IA, sufrieron de *Domain Shift*, rompiendo la síntesis en forma de pitidos robóticos alternos (Efecto R2D2).

### 3. Solución Definitiva: Autoencoder Bi-Canal en el Dominio Complejo
Intentar entrenar una IA independiente para predecir la fase de forma angular es imposible debido al *Phase Wrapping* (las funciones de pérdida lineales colapsan en la frontera de $\pm180^\circ$ al calcular un salto ilusorio de $360^\circ$).

* **Solución:** Desenrollamos el círculo trigonométrico transformando los datos al **Plano Complejo** mediante Coordenadas Cartesianas:
    $$Z = \text{Parte Real }(X) + j \cdot \text{Parte Imaginaria }(Y)$$
* **Implementación:** Un único Autoencoder convolucional procesa simultáneamente la componente **Real** y la **Imaginaria** como si fueran dos canales independientes de color de una imagen (equivalente a capas RGB). Al operar sobre líneas rectas y no sobre ángulos circulares, el gradiente es estable, permitiendo limpiar el ruido y corregir la fase de forma nativa en un solo paso mediante la Transformada Inversa de Fourier Directa (ISTFT).

---

## ⚠️ Limitaciones Actuales (El porqué del tono robótico)

Aunque el Autoencoder Bi-Canal aísla el ruido de fondo con éxito y estabiliza la fase, el audio procesado conserva un leve carácter robótico residual conocido en la industria como **Musical Noise** (Ruido Musical). 

* **Causa técnica:** Las capas `Conv2D` estándar de librerías como Keras/TensorFlow procesan los canales (Real e Imaginario) de manera aislada e independiente. La red no comprende de manera nativa que `X` e `Y` están acopladas por leyes trigonométricas. Ante la incertidumbre de si un píxel es voz o estática, la red tiende a forzar componentes hacia el cero absoluto, generando "agujeros" espectrales rápidos que el oído humano percibe como una voz sintetizada o robótica.

---

## 🔮 Futuras Mejoras (Estado del Arte del Mercado)

Para superar la barrera del *Musical Noise* y lograr una transparencia acústica idéntica a la humana, las siguientes fases de desarrollo se alinearán con los estándares actuales de la investigación:

1.  **Redes Convolucionales Complejas (Deep Complex Networks):** Sustituir las capas tradicionales por operadores de convolución compleja nativa. Esto fuerza a la red a aplicar álgebra de números complejos en los filtros, asegurando que los canales Real e Imaginario se actualicen de manera acoplada y respetando la física de la onda.
2.  **Funciones de Pérdida en el Dominio del Tiempo (SI-SDR / SDR):** En lugar de entrenar a la red obligándola a "mirar" imágenes con MAE/MSE, implementar funciones de pérdida como la *Scale-Invariant Signal-to-Distortion Ratio*. Esto obliga al modelo a optimizar el audio "escuchando" directamente la alineación de la onda resultante en el tiempo.
3.  **Máscaras de Ratio Complejas (Deep Complex Masking):** En lugar de forzar al Autoencoder a generar las coordenadas complejas desde cero (lo cual induce a errores y promedios), la red se rediseñará para predecir una "máscara de filtrado" compleja que se multiplica por el espectrograma ruidoso original, preservando intactos los componentes limpios de la señal.

---

## 🛠️ Requisitos e Instalación

Para ejecutar la interfaz de producción interactiva (`app.py`), asegúrate de contar con las siguientes dependencias instaladas en tu entorno virtual:

```bash
pip install -r requirements.txt
```

Ejecución de la App de Inferencia:
```bash
python app.py
```
