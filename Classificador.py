import cv2
from ultralytics import YOLO

# -- Carregando modelo --
model = YOLO('yolov8n.pt')
#model = YOLO('yolov8m.pt') # Modelo mais preciso (não está rodando)

# -- Aplicando filtros na imagem --
img = cv2.imread('Img/Teste2.jpg')

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

img_processada = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# -- Executando detecção e filtrando direto no YOLO (2: Carro, 3: Moto) --
resultados = model(img_processada, conf=0.25, classes=[2, 3], imgsz=640)[0]

# -- Desenhando as boxes e rótulos --
imagem_anotada = resultados.plot(conf=False, line_width=1)

# -- Exibndo resultado --
cv2.imshow('Detecao de Carros e Motos', imagem_anotada)
cv2.waitKey(0)
cv2.destroyAllWindows()