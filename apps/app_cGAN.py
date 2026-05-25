import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import gradio as gr

# Cargamos el modelo
MODEL_PATH = "models/denoising_cGAN.keras" 
autoencoder = tf.keras.models.load_model(MODEL_PATH)

def limpiar_audio(ruta_audio):

    if ruta_audio is None:
        return None
        
    try:
        # Cargamos el audio completo
        y_completo, sr = librosa.load(ruta_audio, sr=16000)
        chunk_size = 32000  
        audio_limpio_total = [] 
        
        # Bucle para trocear el audio
        for i in range(0, len(y_completo), chunk_size):
            chunk = y_completo[i : i + chunk_size]
            
            if len(chunk) < chunk_size:
                chunk_pad = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")
            else:
                chunk_pad = chunk
                
            stft = librosa.stft(chunk_pad, n_fft=1024, hop_length=256)
            mag = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            fase = np.angle(stft)
            
            min_val, max_val = mag.min(), mag.max()
            mag_norm = (mag - min_val) / (max_val - min_val + 1e-8)
            
            entrada = np.expand_dims(mag_norm, axis=(0, -1))
            pred = autoencoder.predict(entrada, verbose=0)
            img_limpia = np.squeeze(pred)
            
            db_limpio = img_limpia * (max_val - min_val) + min_val
            amp_limpia = librosa.db_to_amplitude(db_limpio)

            stft_rec = amp_limpia * np.exp(1j * fase)
            audio_rec = librosa.istft(stft_rec, hop_length=256)
            
            audio_limpio_total.extend(audio_rec[:len(chunk)])
            
        audio_final = np.array(audio_limpio_total)
        
        # Subimos el volumen porque la frecuencia queda muy baja
        audio_final = audio_final * 100
        
        ruta_salida = "output_limpio_estable.wav"
        sf.write(ruta_salida, audio_final, 16000)
        
        return ruta_salida
        
    except Exception as e:
        return f"Error procesando el audio: {str(e)}"

# --- INTERFAZ WEB CON GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Denoising Autoencoder: Cancelación de Ruido")
    gr.Markdown("Sube un archivo de audio con ruido de fondo o graba directamente con tu micrófono. La Inteligencia Artificial eliminará las frecuencias estáticas.")
    
    with gr.Row():
        with gr.Column():            
            entrada_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="1. Audio Original (Con Ruido)")
            boton_limpiar = gr.Button("✨ Limpiar Audio con IA", variant="primary")
            
        with gr.Column(): 
            salida_audio = gr.Audio(label="2. Audio Procesado (Voz Limpia)", interactive=False)
            
    boton_limpiar.click(fn=limpiar_audio, inputs=entrada_audio, outputs=salida_audio)

if __name__ == "__main__":
    demo.launch()