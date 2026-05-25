import os
import gradio as gr
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

# Parámetros originales de tu transformada de Fourier (STFT)
SAMPLE_RATE = 22050  # Cambia a 16000 si tu primer dataset usaba ese sample rate
N_FFT = 1024         # Genera exactamente 513 bandas (1024 / 2 + 1)
HOP_LENGTH = 256
MAX_PAD_LEN = 126    # El ancho de ventana estricto de tu primer modelo

# Carga tu modelo original (Asegúrate de que la ruta y el nombre coinciden)
autoencoder = tf.keras.models.load_model("models/denoising_autoencoder.keras", compile=False)

# ==========================================
# 2. MOTOR DE LIMPIEZA Y RECONSTRUCCIÓN DE FASE
# ==========================================
def limpiar_audio_lineal(audio_ruidoso_path):
    if audio_ruidoso_path is None:
        return None

    # 1. Cargar el audio original
    y, _ = librosa.load(audio_ruidoso_path, sr=SAMPLE_RATE)

    # 2. Calcular la Magnitud Lineal de la STFT (513 bandas de alto)
    stft_completa = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitud_completa = np.abs(stft_completa)

    # 3. Convertir a Decibelios y Normalizar a [0, 1] 
    # (Ajusta esta fórmula si en tu primer notebook usaste otra normalización)
    mag_db = librosa.amplitude_to_db(magnitud_completa, ref=np.max)
    mag_norm = (mag_db + 80.0) / 80.0
    mag_norm = np.clip(mag_norm, 0.0, 1.0)

    columnas_totales = mag_norm.shape[1]
    tramos_limpios_normalizados = []

    # 4. Trocear y Procesar con el Autoencoder (Ventanas de 126 de ancho)
    for inicio in range(0, columnas_totales, MAX_PAD_LEN):
        fin = inicio + MAX_PAD_LEN
        tramo = mag_norm[:, inicio:fin]

        # Rellenar con ceros (silencio) si el último fragmento se queda corto
        if tramo.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - tramo.shape[1]
            tramo = np.pad(tramo, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Dimensiones para Keras: (1, 513, 126, 1)
        tramo_keras = np.expand_dims(np.expand_dims(tramo, axis=0), axis=-1)

        # Predicción del Autoencoder (Limpia el ruido de fondo visualmente)
        tramo_limpio = autoencoder.predict(tramo_keras, verbose=0)
        tramos_limpios_normalizados.append(np.squeeze(tramo_limpio))

    # 5. Ensamblar todas las magnitudes limpias en una sola matriz gigante
    magnitud_limpia_norm = np.concatenate(tramos_limpios_normalizados, axis=1)
    # Recortamos el exceso de padding final para que dure exactamente lo mismo que el original
    magnitud_limpia_norm = magnitud_limpia_norm[:, :columnas_totales]

    # 6. Deshacer la Normalización matemática
    mag_db_recuperada = (magnitud_limpia_norm * 80.0) - 80.0
    magnitud_final = librosa.db_to_amplitude(mag_db_recuperada)

    # 7. CONSTRUCCIÓN DE LA FASE (Algoritmo Griffin-Lim de alta resolución)
    # Reconstruye la fase de la onda iterando 64 veces sobre la magnitud limpia
    audio_reconstruido = librosa.griffinlim(
        magnitud_final, 
        n_iter=128, 
        hop_length=HOP_LENGTH, 
        win_length=N_FFT,
        momentum=0.99
    )

    audio_reconstruido = audio_reconstruido * 80

    # 8. Guardar el archivo definitivo
    ruta_salida = "audio_limpio_autoencoder_griffin.wav"
    sf.write(ruta_salida, audio_reconstruido, samplerate=SAMPLE_RATE)

    return ruta_salida

# ==========================================
# 3. INTERFAZ GRÁFICA (GRADIO)
# ==========================================
interface = gr.Interface(
    fn=limpiar_audio_lineal,
    inputs=gr.Audio(type="filepath", label="Sube tu archivo de voz con ruido"),
    outputs=gr.Audio(type="filepath", label="Audio procesado (Autoencoder + Griffin-Lim 513)"),
    title="🎙️ Denoising Avanzado por Reconstrucción de Fase",
    description="Este sistema utiliza tu Autoencoder lineal de 513 bandas para limpiar el espectrograma del ruido de fondo, y posteriormente aplica el algoritmo de Griffin-Lim para tejer una fase acústica totalmente nueva y coherente, evitando el uso de la fase corrupta original."
)

if __name__ == "__main__":
    interface.launch()