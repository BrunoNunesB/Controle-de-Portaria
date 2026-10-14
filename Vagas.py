import cv2
import json
import numpy as np

vagas = []
pontos_atuais = []

def clique_mouse(event, x, y, flags, param):
    global pontos_atuais, vagas, img_visualizacao

    if event == cv2.EVENT_LBUTTONDOWN:
        pontos_atuais.append([x, y])
        
        # Ponto onde clicar
        cv2.circle(img_visualizacao, (x, y), 3, (0, 255, 0), -1)

        # Quando completar 4 pontos, desenha o polígono da vaga
        if len(pontos_atuais) == 4:
            pts = np.array(pontos_atuais, np.int32)
            cv2.polylines(img_visualizacao, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            vagas.append(pontos_atuais)
            cv2.putText(img_visualizacao, f"V{len(vagas)}", (pontos_atuais[0][0] + 3, pontos_atuais[0][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            pontos_atuais = []

# 1. Carregar a imagem base
img_original = cv2.imread("Img/Vagas.jpg")
img_visualizacao = img_original.copy()

# 2. Criar a janela uma única vez ANTES do callback
cv2.namedWindow("Marcar Vagas", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Marcar Vagas", clique_mouse)

print("--- MARCADOR DE VAGAS ---")
print("1. Clique nos 4 CANTOS de cada vaga.")
print("2. Pressione 's' para salvar em 'vagas.json'.")
print("3. Pressione 'q' para sair.")

# 3. Loop principal de atualização da janela
while True:
    cv2.imshow("Marcar Vagas", img_visualizacao)
    key = cv2.waitKey(20) & 0xFF
    
    if key == ord('s'):
        with open('vagas.json', 'w') as f:
            json.dump(vagas, f)
        print(f"Sucesso! {len(vagas)} vagas salvas em 'vagas.json'.")
        break
    elif key == ord('q'):
        break

cv2.destroyAllWindows()