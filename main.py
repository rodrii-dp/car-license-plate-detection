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

def detect_text(image, min_confidence=0.5):
    """Detecta texto en una imagen con configuraciones optimizadas"""
    if image is None:
        raise ValueError("No se proporcionó ninguna imagen")

    # Inicializar el lector OCR con configuraciones adicionales
    reader = easyocr.Reader(
        ['en'], 
        gpu=check_cuda_available(),
        model_storage_directory='./models'
    )

    # Obtener dimensiones y redimensionar si es muy grande
    height, width = image.shape[:2]
    max_dimension = 1000
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Detectar texto con parámetros optimizados
    results = reader.readtext(
        processed_image,
        decoder='greedy',
        batch_size=8,
        paragraph=False,
        min_size=10,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4
    )

    # Filtrar resultados por confianza y dibujar
    detected_text = []
    output_image = image.copy()

    for result in results:
        bbox, text, confidence = result

        if confidence > min_confidence:
            # Convertir coordenadas a enteros
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))

            # Dibujar rectángulo y texto
            cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)

            # Mejorar la visualización del texto
            text_position = (top_left[0], max(0, top_left[1] - 10))
            text_to_display = f"{text} ({confidence:.2f})"

            # Añadir fondo negro para mejor legibilidad
            (text_width, text_height), _ = cv2.getTextSize(
                text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                output_image,
                (text_position[0], text_position[1] - text_height),
                (text_position[0] + text_width, text_position[1] + 5),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                output_image, text_to_display, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            detected_text.append(f"{text} (Confianza: {confidence:.2f})")
    
    return output_image, "\n".join(detected_text) if detected_text else "No se detectó ningún texto con suficiente confianza."

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
