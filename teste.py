import cv2
from ultralytics import YOLO

# Carregar modelo pré-treinado YOLOv8 (pode demorar na 1ª vez)
model = YOLO("D:/Faculdade/5º sem - 2025/TCC/TCC - SGD/Youtube Gun Detection/tools/runs/detect/train2/weights/best.pt")  # versão pequena (rápida) do modelo
# Captura de vídeo da webcam
cap = cv2.VideoCapture(0) # use 0 para webcam padrão

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar a imagem da câmera.")
        break

    # Detecção com o YOLO
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Desenhar o retângulo e o texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow("YOLOv8 - Detecção em Tempo Real", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar
cap.release()
cv2.destroyAllWindows()