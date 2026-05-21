import os
import gradio as gr
import tensorflow as tf
import torch
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
MAX_PAD_LEN = 126

# Carga de modelos
generator_cgan = tf.keras.models.load_model("models/denoising_cGAN_mel.keras", compile=False)

# Proceso unificado (cGAN + Vocoder)
def limpiar_audio_completo(audio_ruidoso_path):
    if audio_ruidoso_path is None:
        return None
    
    # 1. Cargar el audio
    y, _ = librosa.load(audio_ruidoso_path, sr=SAMPLE_RATE)
    
    # 2. Generar Mel completo
    mel_spec_completo = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db_completo = librosa.power_to_db(mel_spec_completo, ref=np.max)
    mel_norm_completo = (mel_db_completo + 80.0) / 80.0
    
    columnas_totales = mel_norm_completo.shape[1]
    audio_tramos_limpios = []
    
    # 3. Procesar por fragmentos
    for inicio in range(0, columnas_totales, MAX_PAD_LEN):
        fin = inicio + MAX_PAD_LEN
        tramo = mel_norm_completo[:, inicio:fin]
        
        if tramo.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - tramo.shape[1]
            tramo = np.pad(tramo, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        tramo_keras = np.expand_dims(np.expand_dims(tramo, axis=0), axis=-1)
        
        # --- LIMPIEZA CON TU cGAN ---
        tramo_limpio_pred = generator_cgan.predict(tramo_keras, verbose=0)
        
        # --- SÍNTESIS CON GRIFFIN-LIM (LIBROSA) ---
        mel_squeezed = np.squeeze(tramo_limpio_pred) 
        
        # 1. Volvemos a Decibelios
        mel_db = (mel_squeezed * 80.0) - 80.0
        
        # 2. Pasamos de Decibelios a Energía pura
        mel_energia = librosa.db_to_power(mel_db)
        
        # 3. Librosa hace la magia matemática para recuperar la onda de audio
        onda_tramo = librosa.feature.inverse.mel_to_audio(
            mel_energia, 
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
            
        audio_tramos_limpios.append(onda_tramo)

    # Pegamos los fragmentos
    audio_final = np.concatenate(audio_tramos_limpios)
    
    # Guardamos (Librosa mantiene los 22050 Hz de tu audio original)
    ruta_salida = "audio_limpio_griffinlim.wav"
    sf.write(ruta_salida, audio_final, samplerate=SAMPLE_RATE)
    
    return ruta_salida

# Interfaz
interface = gr.Interface(
    fn=limpiar_audio_completo,
    inputs=gr.Audio(type="filepath", label="Subir Audio con Ruido (Cualquier duración)"),
    outputs=gr.Audio(type="filepath", label="Audio Limpio (Voz Natural - HiFi-GAN)"),
    title="🎙️ Denoising Avanzado (cGAN + Vocoder)",
    description="Sube un archivo de audio de cualquier duración. El sistema lo procesará en fragmentos utilizando la cGAN para la limpieza y HiFi-GAN para la síntesis fotorrealista."
)

if __name__ == "__main__":
    interface.launch()