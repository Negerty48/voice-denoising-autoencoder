import os
import gradio as gr
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO BI-CANAL
# ==========================================
print("⏳ Cargando tu Autoencoder Bi-Canal definitivo...")

# Parámetros originales de tu transformada de Fourier
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
MAX_PAD_LEN = 126

# Cargar el modelo final entrenado con números complejos
autoencoder = tf.keras.models.load_model("models/denoising_complejo_final.keras", compile=False)

print("✅ Modelo cargado con éxito. El servidor está listo.")

# ==========================================
# 2. MOTOR DE PROCESAMIENTO MATEMÁTICO (ISTFT)
# ==========================================
def limpiar_audio_complejo_app(audio_ruidoso_path):
    if audio_ruidoso_path is None:
        return None

    # 1. Cargar el audio
    y, _ = librosa.load(audio_ruidoso_path, sr=SAMPLE_RATE)

    # 2. Calcular la STFT (Matriz de números complejos cruda)
    stft_completa = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 3. Normalización Maestra
    # Guardamos el valor máximo para poder deshacer la escala al final
    max_val = np.max(np.abs(stft_completa)) + 1e-9
    stft_norm = stft_completa / max_val

    columnas_totales = stft_norm.shape[1]
    tramos_limpios_complejos = []

    # 4. Procesamiento por fragmentos
    for inicio in range(0, columnas_totales, MAX_PAD_LEN):
        fin = inicio + MAX_PAD_LEN
        tramo = stft_norm[:, inicio:fin]

        # Rellenar con ceros si es el último tramo
        if tramo.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - tramo.shape[1]
            tramo = np.pad(tramo, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 5. Separar Real e Imaginario y apilar en 2 canales
        parte_real = np.real(tramo)
        parte_imaginaria = np.imag(tramo)
        tramo_keras = np.stack([parte_real, parte_imaginaria], axis=-1) # (513, 126, 2)
        tramo_keras = np.expand_dims(tramo_keras, axis=0)               # (1, 513, 126, 2)

        # 6. --- INFERENCIA DE LA RED NEURONAL ---
        tramo_pred = autoencoder.predict(tramo_keras, verbose=0)
        
        # 7. Recuperar los canales predichos
        pred_real = tramo_pred[0, :, :, 0]
        pred_imag = tramo_pred[0, :, :, 1]
        
        # 8. Reconstruir el número complejo (Z = a + bi)
        # En Python, '1j' representa la unidad imaginaria 'i'
        tramo_complejo = pred_real + 1j * pred_imag
        tramos_limpios_complejos.append(tramo_complejo)

    # 9. Ensamblar la matriz gigante limpia
    stft_limpia_norm = np.concatenate(tramos_limpios_complejos, axis=1)
    stft_limpia_norm = stft_limpia_norm[:, :columnas_totales]

    # 10. Deshacer la normalización
    stft_final = stft_limpia_norm * max_val

    # 11. --- TRANSFORMADA INVERSA DIRECTA (ISTFT) ---
    # Adiós Vocoders, adiós Griffin-Lim. Matemática pura y exacta.
    print("🪄 Aplicando ISTFT (Transformada Inversa de Fourier)...")
    audio_limpio = librosa.istft(stft_final, hop_length=HOP_LENGTH, n_fft=N_FFT)

    # 12. Guardar el archivo definitivo
    ruta_salida = "audio_definitivo_bicanal.wav"
    sf.write(ruta_salida, audio_limpio, samplerate=SAMPLE_RATE)
    print(f"🎉 Audio impecable guardado en: {ruta_salida}")

    return ruta_salida

# ==========================================
# 3. INTERFAZ GRÁFICA (GRADIO)
# ==========================================
interface = gr.Interface(
    fn=limpiar_audio_complejo_app,
    inputs=gr.Audio(type="filepath", label="Sube tu archivo de voz con ruido"),
    outputs=gr.Audio(type="filepath", label="Audio procesado (Autoencoder Bi-Canal)"),
    title="🎙️ Denoising de Grado de Investigación (Dominio Complejo)",
    description="Sistema de mejora de voz de vanguardia. Utiliza una arquitectura de red neuronal de doble canal que procesa simultáneamente las componentes Real e Imaginaria del espectrograma, permitiendo una reconstrucción acústica y de fase matemáticamente perfecta sin depender de heurísticas o vocoders.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch()