import cv2
from ultralytics import YOLO
import winsound  # Apenas no Windows

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")  # Certifique-se de que esse caminho esteja correto

# Abre a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Loop para capturar quadros da câmera
while True:
    # Lê um frame (uma imagem do vídeo) da câmera
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        break

    # Executa a detecção de objetos no frame
    results = model(frame)

    # Verifica se a detecção encontrou objetos relevantes
    for result in results:
        for obj in result.boxes:
            # Verifica se a classe detectada é faca ou arma
            # Ajuste os números abaixo de acordo com as classes do seu modelo
            if obj.cls in [43, 1]:  # Ajuste os índices de acordo com suas classes

                # Acesse as coordenadas da caixa delimitadora corretamente
                x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Verifique se `obj.xyxy` retorna uma lista

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "FACA/ARMA DETECTADA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Soa um alarme
                winsound.Beep(2000, 500)  # Som de alarme no Windows

    # Mostra o frame na tela
    cv2.imshow("Câmera ao Vivo", frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha a câmera e a janela
cap.release()
cv2.destroyAllWindows()
