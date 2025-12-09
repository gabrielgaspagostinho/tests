import cv2
import numpy as np
import argparse

# --- 1. A Função de Letterbox (Pré-processamento) ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Redimensiona a imagem mantendo a proporção (aspect ratio) e adicionando bordas (padding).
    Retorna a imagem redimensionada, a taxa de escala (ratio) e o padding (dw, dh).
    """
    shape = im.shape[:2]  # formato atual [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calcula a taxa de escala (o menor lado define a escala para não cortar nada)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Apenas reduz se scaleup for False (opcional)
    if not scaleup:
        r = min(r, 1.0)

    # Calcula o tamanho sem padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Calcula o padding necessário para chegar em 640x640
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding wh

    if auto:  # mínimo retângulo (útil para treinar, menos para inferência estática ONNX)
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:  # esticar (desabilita letterbox)
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # Divide o padding por 2 para centralizar a imagem
    dw /= 2
    dh /= 2

    # Redimensiona a imagem se necessário
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Adiciona as bordas cinzas
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)

class InferenceYOLO():
    def __init__(self, MODEL_PATH, IMAGE_PATH, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD):
        # 1. Carregar Modelo ONNX
        print(f"Carregando modelo: {MODEL_PATH}")
        try:
            net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return

        # 2. Carregar Imagem Original
        self.original_image = cv2.imread(IMAGE_PATH)
        if self.original_image is None:
            print(f"Erro ao abrir imagem: {IMAGE_PATH}")
            return

        # 3. PRÉ-PROCESSAMENTO COM LETTERBOX (A Mágica)
        # Transforma qualquer tamanho em 640x640 sem distorção
        # 'ratio' e 'dwdh' são fundamentais para mapear as coordenadas de volta depois
        input_image, ratio, (dw, dh) = letterbox(self.original_image, new_shape=INPUT_SIZE, auto=False)

        # Cria o Blob para o OpenCV (Normaliza 0-255 para 0-1 e ajusta canais)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)

        # 4. Inferência
        outputs = net.forward()

        # 5. Pós-processamento com NumPy
        # Transpõe a saída: (1, 8400, 5) -> (Batch, Detecções, Dados)
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        self.boxes = []
        self.scores = []
        self.class_ids = []

        self.cleanboxes = []
        self.cleanscores = []
        self.cleanclass_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)

            if max_score >= SCORE_THRESHOLD:
                row = outputs[0][i]

                # --- COORDENADAS NO MUNDO LETTERBOX (640x640) ---
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                # --- A MATEMÁTICA DE VOLTA (Desfazendo o Letterbox) ---
                # 1. Removemos o padding (subtraímos as bordas cinzas)
                cx = cx - dw
                cy = cy - dh
                w = w  # Largura e altura não mudam com translação, só com escala
                h = h

                # 2. Revertemos a escala (dividimos pelo ratio)
                cx = cx / ratio[0]
                cy = cy / ratio[1]
                w = w / ratio[0]
                h = h / ratio[1]

                # 3. Converte Centro para Canto Superior Esquerdo (para o OpenCV desenhar)
                left = int(cx - w / 2)
                top = int(cy - h / 2)
                width = int(w)
                height = int(h)

                self.boxes.append([left, top, width, height])
                self.scores.append(float(max_score))
                self.class_ids.append(max_class_loc[1])

        # 6. Non-Maximum Suppression (Limpeza)
        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.scores, SCORE_THRESHOLD, NMS_THRESHOLD)

        if len(self.indices) > 0:
            print(f"Detectados {len(self.indices)} objetos.")
            for i in self.indices.flatten():
                self.cleanboxes.append(self.boxes[i])
                self.cleanscores.append(self.scores[i])
                self.cleanclass_ids.append(self.class_ids[i])

    def imgshow(self):
        # 7. Desenhar na Imagem ORIGINAL
        if len(self.indices) > 0:
            print(f"Detectados {len(self.indices)} objetos.")
            for i in self.indices.flatten():
                box = self.boxes[i]
                self.x, self.y, self.w, self.h = box[0], box[1], box[2], box[3]
                score = self.scores[i]

                # Desenha retângulo
                cv2.rectangle(self.original_image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)

                # Texto
                label = f"Classe {self.class_ids[i]}: {score:.2f}"
                cv2.putText(self.original_image, label, (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("Nenhum objeto detectado.")

        # Mostra resultado
        # Redimensiona a janela se a imagem for muito grande (apenas para visualização)
        display_img = self.original_image
        if self.original_image.shape[0] > 1000:
            display_img = cv2.resize(self.original_image, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Inferencia YOLO Letterbox", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bestboxes(self):
        return self.cleanboxes

    def bestclassids(self):
        return self.cleanclass_ids

    def bestscores(self):
        return self.cleanscores


class InferenceClass():
    def __init__(self,MODEL_PATH, IMAGE_PATH, INPUT_SIZE, CLASS_NAMES):
        self.class_names = CLASS_NAMES
        def softmax(x):
            """Função auxiliar para transformar números brutos em porcentagens (0 a 1)"""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # 1. Carregar a Rede Neural
        try:
            net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar modelo (verifique se exportou com opset=12): {e}")

        # 2. Ler a Imagem
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            print("Imagem não encontrada.")

        # 3. Pré-processamento
        input_image, ratio, (dw, dh) = letterbox(self.image, new_shape=INPUT_SIZE, auto=False)

        # Cria o Blob para o OpenCV (Normaliza 0-255 para 0-1 e ajusta canais)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)

        net.setInput(blob)

        # 4. Inferência
        # Retorna uma matriz (1, Num_Classes)
        outputs = net.forward()

        # 5. Pós-processamento (Muito simples!)
        scores = outputs[0]  # Pega a primeira linha (nosso batch é 1)

        # Aplica softmax para ter certeza que somam 1.0 (opcional, mas bom para ler %)
        self.probs = softmax(scores)

        # Encontra o índice da maior probabilidade
        self.class_id = np.argmax(self.probs)
        self.max_score = self.probs[self.class_id]
        self.class_name = CLASS_NAMES[self.class_id] if self.class_id < len(CLASS_NAMES) else str(self.class_id)

        # 6. Mostrar Resultado
        print(f"\n--- RESULTADO DA CLASSIFICAÇÃO ---")
        print(f"Classe Predita: {self.class_name}")
        print(f"ID: {self.class_id}")
        print(f"Confiança: {self.max_score:.2f} ({self.max_score * 100:.1f}%)")

        # Imprime todas as probabilidades
        print("\nDetalhes:")
        for i, prob in enumerate(self.probs):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            print(f"  - {i}, {name}: {prob:.4f}")

    def classresult(self):
        return [int(self.class_id), round(float(self.max_score), 2), self.class_name]

    def imgshow(self):
        # Desenha na imagem apenas o texto (não tem caixa/box em classificação!)
        text = f"{self.class_name}: {self.max_score:.2f}"
        cv2.putText(self.image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Classificacao YOLO", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


sla = InferenceClass("runs/classify/yolo_manometer_legivel/weights/best.onnx", "manometroilegivel.jpg", (640, 640), ['ilegivel', 'legivel'])
print(sla.classresult())
sla.imgshow()

yolo = InferenceYOLO("runs/detect/yolov8n_platform_detector10/weights/best.onnx", "image10.png",(640,640), 0.5, 0.45)
print(yolo.bestboxes())
print(yolo.bestscores())
yolo.imgshow()

# 448x448