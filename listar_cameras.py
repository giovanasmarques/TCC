import cv2

def aberta(i):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # no Windows, ajuda a abrir mais rápido
    ok = cap.isOpened()
    cap.release()
    return ok

print("Testando índices 0..4")
for i in range(5):
    print(f"- Índice {i}: ", "DISPONÍVEL" if aberta(i) else "vazio")
