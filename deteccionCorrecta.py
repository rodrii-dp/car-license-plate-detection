import cv2 #para procesar los videos usamos cv2, y redimnesionarlos para optimizar el procesamiento
import torch
import pytesseract  #lo usamos para extraer el texto de las matriculas
import gradio as gr  #para la interfaz grafica
from ultralytics import YOLO    #modelo usado para la deteccion de vehiculos
import time      #control del tiempo que tarda
import re  #para usar expresiones regulares para las matriculas, detectando patrones por pais

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#defino los patrones de matrículas por país
PAISES_Y_PATRONES = {
    "España": r'\b[A-Z0-9]{4}[A-Z]{3}\b',
    "Alemania": r'\b[A-Z]{1,3}[0-9]{1,4}\s?[A-Z]{1,3}\b',
    "Reino Unido": r'\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b',
}

#cargo el modelo YOLOv8 una sola vez y utilizar la GPU si está disponible con la libreria torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilizando dispositivo: {device}")
try:
    model = YOLO(
        'yolov8n.pt')
    model.to(device)
except Exception as e:
    print(f"Error al cargar el modelo YOLOv8: {e}")
    exit(1)


def procesar_texto_matricula(texto):
    try:

        texto_procesado = texto.replace(" ", "").upper()  #eliminamos los espacios vacios y pasamos a mayusculas todas las letras

        for pais, patron in PAISES_Y_PATRONES.items():
            match = re.search(patron, texto_procesado)
            if match:
                print(f"Matrícula detectada: {match.group(0)} - País: {pais}")
                return match.group(0)   #retornamos la matricula detectada

        return None
    except Exception as e:
        print(f"Error en procesar_texto_matricula: {e}")
        return None

def detectar_matricula(vehiculo):
    try:
        #preprocesamiento para mejorar el OCR
        gray = cv2.cvtColor(vehiculo, cv2.COLOR_BGR2GRAY)  #aplicamos una escala de grises a la imagen para mejorar el preprocesamiento de OCR
        blur = cv2.medianBlur(gray, 3) #y el filtro  blur  igual
        #aplico OCR
        texto = pytesseract.image_to_string(blur, config='--psm 8')
        #proceso el texto para extraer la matrícula
        matricula = procesar_texto_matricula(texto)
        return matricula  #devuelvo la matricula extraida
    except Exception as e:
        print(f"Error en detectar_matricula: {e}")
        return None


def clasificar_vehiculo(label):
    if label in ['truck', 'bus', 'van']:  #pasamos todas las clases que no sean coche a la etiuqeta furgoneta, ya que solo se permite coche
        return 'furgoneta'
    else:
        return 'coche'


def generar_resultados(lista_matriculas):
    try:
        if not lista_matriculas:
            return [["No se detectaron matrículas", "", ""]]  #si no hay matriculas te avisa
        resultados = []
        for item in lista_matriculas:  #itera sobre la lista de matriculas
            estado = "Incumpliendo" if item['tipo'] == 'furgoneta' else "Permitido"  #si es furgoneta muestra que esta incumpliendo la normativa y si es coche esta permitido
            resultados.append([item['matricula'], item['tipo'], estado])
        return resultados
    except Exception as e:
        print(f"Error en generar_resultados: {e}")
        return [["Error al generar resultados", "", ""]]


def procesar_video(video):
    try:

        if isinstance(video, dict):
            video_path = video['name']
        else:
            video_path = video

        #inicializo la captura de video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error al abrir el video.")
            return [["Error al abrir el video", "", ""]]

        lista_matriculas = []

        #obtengo información del video como el total de fotogramas por segundo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Total de fotogramas en el video: {total_frames}, FPS: {fps}")

        #optimizo a 5 los fotogramas que procesa por segundo para que no tarde mucho en hacer el procesado ya que si no estaria mucho tiempo
        process_every_n_frames = 5  #procesamos cada 5 fotogramas
        resize_width = 640  #ancho al que se redimensionará el video
        resize_height = 360  #alto al que se redimensionará el video

        frame_count = 0
        processed_frames = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n_frames == 0:
                #redimensiono el fotograma para acelerar el procesamiento
                frame_resized = cv2.resize(frame, (resize_width, resize_height))

                #realizo detecciones en el frame redimensionado
                try:
                    results = model(frame_resized)
                except Exception as e:
                    print(f"Error al realizar detecciones: {e}")
                    continue

                #proceso las detecciones
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        try:
                            #obtengo las coordenadas y la etiqueta de la detección
                            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            label = model.names[cls]

                            if label in ['car', 'truck', 'bus', 'van']:
                                #recorto el vehículo detectado
                                vehiculo = frame_resized[ymin:ymax, xmin:xmax]
                                #detecto  la matrícula
                                matricula = detectar_matricula(vehiculo)
                                if matricula:
                                    tipo = clasificar_vehiculo(label)
                                    lista_matriculas.append({'matricula': matricula, 'tipo': tipo})

                                    print(f"Matrícula detectada: {matricula} - Tipo: {tipo}")
                        except Exception as e:
                            print(f"Error al procesar una detección: {e}")
                            continue

                processed_frames += 1

                #para comprobar que esta llendo en orden imprimo por la terminal  los fotogramas procesados
                if processed_frames % 10 == 0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"Procesados {processed_frames} fotogramas cada {process_every_n_frames} frames en {elapsed_time:.2f} segundos")

            frame_count += 1

        cap.release()
        total_time = time.time() - start_time
        print(f"Procesamiento completado en {total_time:.2f} segundos")  #muestro el tiempo que ha tardado en procesarlo
        return generar_resultados(lista_matriculas)

    except Exception as e:
        print(f"Error en procesar_video: {e}")
        return [["Error durante el procesamiento del video", "", ""]]


# interfaz de gradio que muestra la tabla en la salida ddel procesamiento tambien
iface = gr.Interface(
    fn=procesar_video,
    inputs=gr.Video(),
    outputs=gr.Dataframe(headers=["Matrícula", "Tipo", "Estado"]),
    title="Detección de Matrículas y Clasificación de Vehículos",
    description="Sube un video para detectar las matrículas y clasificar los vehículos entre furgonetas y coches."
)

if __name__ == "__main__":
    iface.launch()