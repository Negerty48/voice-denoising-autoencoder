import os
import gradio as gr
import tensorflow as tf
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import SpeechT5HifiGan

# ==========================================
# 1. CONFIGURACIÓN EXACTA DE HIFI-GAN
# ==========================================
print("⏳ Cargando modelos en memoria... Por favor, espera.")

# Parámetros estrictos de Microsoft
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0
MAX_PAD_LEN = 128

# Constantes de desnormalización Log-Mel
MIN_MEL = -11.5129
MAX_MEL = 2.0
RANGO_MEL = MAX_MEL - MIN_MEL

# Cargar tus modelos definitivos
generator_cgan = tf.keras.models.load_model("models/denoising_cGAN_hifigan.keras", compile=False)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

print("✅ Modelos cargados. Servidor listo para procesar audios.")

# ==========================================
# 2. MOTOR DE PROCESAMIENTO CON IA
# ==========================================
def limpiar_audio_completo(audio_ruidoso_path):
    if audio_ruidoso_path is None:
        return None

    # 1. Cargar el audio forzando los 16kHz
    y, _ = librosa.load(audio_ruidoso_path, sr=SAMPLE_RATE)

    # 2. Calcular Espectrograma Log-Mel con matemática exacta
    magnitud = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    mel_spec = np.dot(mel_basis, magnitud)
    mel_log_completo = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))

    columnas_totales = mel_log_completo.shape[1]
    audio_tramos_limpios = []

    # 3. Trocear y Procesar en bucle
    for inicio in range(0, columnas_totales, MAX_PAD_LEN):
        fin = inicio + MAX_PAD_LEN
        tramo_log = mel_log_completo[:, inicio:fin]

        # Rellenar con "silencio" si el último tramo es más corto de 128
        if tramo_log.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - tramo_log.shape[1]
            tramo_log = np.pad(tramo_log, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=MIN_MEL)

        # 4. Normalizar a [0, 1] para tu cGAN
        tramo_norm = (tramo_log - MIN_MEL) / RANGO_MEL
        tramo_norm = np.clip(tramo_norm, 0.0, 1.0)
        
        # Expandir dimensiones para Keras: (1, 80, 128, 1)
        tramo_keras = np.expand_dims(np.expand_dims(tramo_norm, axis=0), axis=-1)

        # --- FASE IA 1: LIMPIEZA CON TU cGAN ---
        tramo_limpio_pred = generator_cgan.predict(tramo_keras, verbose=0)
        
        # --- FASE IA 2: SÍNTESIS CON EL VOCODER NEURONAL ---
        mel_squeezed = np.squeeze(tramo_limpio_pred) # Matriz plana de (80, 128)
        
        # Deshacemos la normalización para recuperar el Log-Mel exacto
        mel_log_recuperado = (mel_squeezed * RANGO_MEL) + MIN_MEL
        
        # Transponemos para Hugging Face (Tiempo, Frecuencias) -> (128, 80)
        mel_tensor = torch.tensor(mel_log_recuperado.T).unsqueeze(0).float()
        
        # El Vocoder inventa la fase y reconstruye la onda física
        with torch.no_grad():
            onda_tramo = vocoder(mel_tensor)
            
        audio_tramos_limpios.append(onda_tramo.squeeze().numpy())

    # 5. Ensamblar los vagones del audio
    audio_final = np.concatenate(audio_tramos_limpios)
    audio_final = audio_final * 80

    # 6. Guardar archivo final
    ruta_salida = "audio_limpio_ia_definitivo.wav"
    sf.write(ruta_salida, audio_final, samplerate=SAMPLE_RATE)

    return ruta_salida

# ==========================================
# 3. INTERFAZ GRÁFICA (GRADIO)
# ==========================================
interface = gr.Interface(
    fn=limpiar_audio_completo,
    inputs=gr.Audio(type="filepath", label="Sube tu archivo de voz con ruido"),
    outputs=gr.Audio(type="filepath", label="Audio procesado (cGAN + Vocoder Neuronal)"),
    title="🎙️ Eliminación de Ruido de Calidad Industrial",
    description="Sube un archivo de audio. Este sistema utiliza una red generativa adversarial condicional (cGAN) para aislar la frecuencia vocal en tiempo real, seguida de un vocoder neuronal HiFi-GAN que sintetiza la voz humana de manera fotorrealista eliminando la fase matemática clásica.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch()