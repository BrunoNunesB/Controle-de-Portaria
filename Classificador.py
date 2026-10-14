import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
if not os.path.exists('vagas.json'):
    raise FileNotFoundError("Arquivo 'vagas.json' não encontrado! Execute 'Vagas.py' primeiro.")

with open('vagas.json', 'r') as f:
    vagas = json.load(f)

# 2. Carregar imagem de teste
caminho_imagem = 'Img/Teste1.jpg'
img = cv2.imread(caminho_imagem)

# Executar YOLO
resultados = model(img, conf=0.10, imgsz=1024)[0]
veiculos = resultados.boxes.xyxy.cpu().numpy() if resultados.boxes is not None else []

imagem_resultado = img.copy()
total_livres, total_ocupadas = 0, 0

# 3. Testar cada vaga
for idx, vaga_pts in enumerate(vagas):
    pts = np.array(vaga_pts, np.int32)
    esta_ocupada = False

    for v in veiculos:
        cx_carro = float((v[0] + v[2]) / 2)
        cy_carro = float((v[1] + v[3]) / 2)
        
        dist = cv2.pointPolygonTest(pts, (cx_carro, cy_carro), False)
        if dist >= 0:
            esta_ocupada = True
            break

    if not esta_ocupada:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        recorte_pixels = img[mask == 255]
        if recorte_pixels.size > 0 and np.std(recorte_pixels) > 35.0:
            esta_ocupada = True

    if esta_ocupada:
        cor = (0, 0, 255)
        total_ocupadas += 1
    else:
        cor = (0, 255, 0)
        total_livres += 1

    cv2.polylines(imagem_resultado, [pts], isClosed=True, color=cor, thickness=2)
    cv2.putText(imagem_resultado, f"V{idx+1}", (vaga_pts[0][0] + 3, vaga_pts[0][1] + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

# Painel Informativo
cv2.rectangle(imagem_resultado, (10, 10), (200, 60), (0, 0, 0), -1)
cv2.putText(imagem_resultado, f"Livres: {total_livres}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(imagem_resultado, f"Ocupadas: {total_ocupadas}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Monitoramento de Vagas", imagem_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()