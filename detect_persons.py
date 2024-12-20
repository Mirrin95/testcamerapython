import cv2

# Cargar el modelo preentrenado para la detección de personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Ruta del video local
video_path = 'videooo.mp4'  # El nombre del archivo de video es correcto

# Abrir el video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar personas en el frame con parámetros ajustados
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Dibujar rectángulos alrededor de las personas detectadas
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow('Person Detection', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
