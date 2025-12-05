from ultralytics import YOLO

model = YOLO("D:/Faculdade/5º sem - 2025/TCC/TCC - SGD/Youtube Gun Detection/tools/runs/detect/train2/weights/best.pt")  # versão pequena (rápida) do modelo
model.predict(source="D:/Faculdade/5º sem - 2025/TCC/TCC - SGD/Youtube Gun Detection/tools/images/val/oRGDT81hbg8_30_60_000147.jpg", show=True, conf=0.01)

print(model.names)

# import cv2
# import os

# # Caminhos
# image_dir = "D:/Faculdade/5º sem - 2025/TCC/TCC - SGD/Youtube Gun Detection/tools/images/train"
# label_dir = "D:/Faculdade/5º sem - 2025/TCC/TCC - SGD/Youtube Gun Detection/tools/labels/train"

# # Lista todas as imagens
# for img_file in os.listdir(image_dir):
#     if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#         img_path = os.path.join(image_dir, img_file)
#         label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')

#         # Carrega a imagem
#         img = cv2.imread(img_path)
#         h, w, _ = img.shape

#         # Se tiver label, desenha as caixas
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     cls, x_c, y_c, bw, bh = map(float, line.strip().split())
#                     x1 = int((x_c - bw / 2) * w)
#                     y1 = int((y_c - bh / 2) * h)
#                     x2 = int((x_c + bw / 2) * w)
#                     y2 = int((y_c + bh / 2) * h)

#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(img, str(int(cls)), (x1, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Mostra a imagem
#         cv2.imshow("Labels", img)
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break

# cv2.destroyAllWindows()
