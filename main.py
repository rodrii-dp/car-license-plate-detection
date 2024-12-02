import cv2
import easyocr
import numpy as np
import gradio as gr
from pathlib import Path

def check_cuda_available():
    """Verifica si CUDA está disponible"""
    try:
        cv2.cuda.deviceCount()
        return True
    except:
        return False

def preprocess_image(image):
    """Preprocesa la imagen para mejorar la detección de texto"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque gaussiano para reducir ruido
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Aplicar umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Aplicar operaciones morfológicas para limpiar la imagen
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

def process_image(input_image, confidence_threshold):
    """Función para procesar la imagen desde Gradio"""
    try:
        # Convertir la imagen de Gradio a formato OpenCV
        image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Procesar la imagen
        output_image, detected_text = detect_text(image, min_confidence=confidence_threshold)
        
        # Convertir la imagen de vuelta a RGB para Gradio
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
        return output_image, detected_text
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Crear la interfaz de Gradio
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Subir imagen"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Umbral de confianza")
    ],
    outputs=[
        gr.Image(label="Imagen procesada"),
        gr.Textbox(label="Texto detectado")
    ],
    title="Detector de Texto en Imágenes",
    description="Sube una imagen para detectar y reconocer texto. Ajusta el umbral de confianza según sea necesario."
)

# Iniciar la aplicación
if __name__ == "__main__":
    iface.launch(share=True)
