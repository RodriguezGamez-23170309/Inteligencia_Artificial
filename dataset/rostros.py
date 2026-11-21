import cv2, os
#Omar Ernesto Rodriguez Gamez
ENTRADA = "fotos_originales"
SALIDA = "dataset_rostros"
TAM = (160, 160)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for persona in os.listdir(ENTRADA):
    carpeta_in = os.path.join(ENTRADA, persona)
    carpeta_out = os.path.join(SALIDA, persona)
    os.makedirs(carpeta_out, exist_ok=True)

    for i, archivo in enumerate(os.listdir(carpeta_in)):
        img = cv2.imread(os.path.join(carpeta_in, archivo))
        if img is None:
            continue
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rostros = detector.detectMultiScale(gris, 1.3, 5)
        for (x, y, w, h) in rostros[:1]:      # solo el primer rostro
            rostro = cv2.resize(img[y:y+h, x:x+w], TAM)
            cv2.imwrite(os.path.join(carpeta_out, f"{persona}_{i}.jpg"), rostro)

print("Listo")