import cv2
import numpy as np


# --- 1. A Função de Letterbox (Pré-processamento) ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Redimensiona a imagem mantendo a proporção (aspect ratio) e adicionando bordas (padding).
    """
    shape = im.shape[:2]  # formato atual [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)


# --- 2. Classe para a box de uma detecção ---
class DetectionBox:
    def __init__(self, x, y, w, h, score, class_id):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.score = float(score)
        self.class_id = int(class_id)

    def __repr__(self):
        # Representação em string para facilitar o print
        return f"Box(Class: {self.class_id}, Score: {self.score:.2f}, XYWH: [{self.x}, {self.y}, {self.w}, {self.h}])"


# --- 3. Classe de Inferência de detecção (YOLO) ---
class YOLODetection():
    def __init__(self, MODEL_PATH, IMAGE_PATH, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD):
        # Carregar Modelo
        print(f"Carregando modelo DETECT: {MODEL_PATH}")
        try:
            self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.detections = []
            return

        # Carregar Imagem
        self.original_image = cv2.imread(IMAGE_PATH)
        if self.original_image is None:
            print(f"Erro ao abrir imagem: {IMAGE_PATH}")
            self.detections = []
            return

        # Pré-processamento
        input_image, ratio, (dw, dh) = letterbox(self.original_image, new_shape=INPUT_SIZE, auto=False)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
        self.net.setInput(blob)

        # Inferência
        outputs = self.net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        # Listas TEMPORÁRIAS apenas para o cálculo do NMS
        # O OpenCV NMS exige listas separadas
        temp_boxes = []
        temp_scores = []
        temp_class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)

            if max_score >= SCORE_THRESHOLD:
                row = outputs[0][i]
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                # Reverter Letterbox
                cx = (cx - dw) / ratio[0]
                cy = (cy - dh) / ratio[1]
                w = w / ratio[0]
                h = h / ratio[1]

                left = int(cx - w / 2)
                top = int(cy - h / 2)
                width = int(w)
                height = int(h)

                temp_boxes.append([left, top, width, height])
                temp_scores.append(float(max_score))
                temp_class_ids.append(max_class_loc[1])

        # NMS (Limpeza)
        indices = cv2.dnn.NMSBoxes(temp_boxes, temp_scores, SCORE_THRESHOLD, NMS_THRESHOLD)

        # --- AQUI ESTÁ A MUDANÇA PRINCIPAL ---
        # Criamos uma lista única de objetos DetectionBox
        self.detections = []  # Lista de objetos

        if len(indices) > 0:
            print(f"Detectados {len(indices)} objetos.")
            for i in indices.flatten():
                # Instancia o objeto Box
                box_obj = DetectionBox(
                    x=temp_boxes[i][0],
                    y=temp_boxes[i][1],
                    w=temp_boxes[i][2],
                    h=temp_boxes[i][3],
                    score=temp_scores[i],
                    class_id=temp_class_ids[i]
                )
                self.detections.append(box_obj)
        else:
            print("Nenhum objeto detectado.")

    def imgshow(self):
        if not hasattr(self, 'original_image') or self.original_image is None: return

        # Itera sobre os OBJETOS, não sobre índices soltos
        for box in self.detections:
            # Acessa os atributos do objeto diretamente
            cv2.rectangle(self.original_image, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2)
            label = f"ID {box.class_id}: {box.score:.2f}"
            cv2.putText(self.original_image, label, (box.x, box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Redimensiona para visualização se for muito grande
        display_img = self.original_image
        if self.original_image.shape[0] > 1000:
            display_img = cv2.resize(self.original_image, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Inferencia YOLO (Objetos Box)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_detections(self):
        """Retorna a lista de objetos DetectionBox"""
        return self.detections


# --- 4. Classe de Inferência de Classificação ---
class YOLOClass():
    def __init__(self, MODEL_PATH, IMAGE_PATH, INPUT_SIZE, CLASS_NAMES):
        self.class_names = CLASS_NAMES

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        print(f"Carregando modelo CLS: {MODEL_PATH}")
        try:
            self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return

        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            print("Imagem não encontrada.")
            return

        input_image, ratio, (dw, dh) = letterbox(self.image, new_shape=INPUT_SIZE, auto=False)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, INPUT_SIZE, swapRB=True,
                                     crop=False)  # crop=True para CLS geralmente

        self.net.setInput(blob)
        outputs = self.net.forward()

        scores = outputs[0]
        self.probs = softmax(scores)
        self.class_id = np.argmax(self.probs)
        self.max_score = self.probs[self.class_id]
        self.class_name = CLASS_NAMES[self.class_id] if self.class_id < len(CLASS_NAMES) else str(self.class_id)

        print(f"Classe Predita: {self.class_name} ({self.max_score:.2f})")

    def classresult(self):
        # Retorna um dicionário ou objeto simples (Classificação não tem 'Box')
        return {
            "id": int(self.class_id),
            "score": round(float(self.max_score), 2),
            "name": self.class_name
        }

    def imgshow(self):
        if not hasattr(self, 'image') or self.image is None: return
        text = f"{self.class_name}: {self.max_score:.2f}"
        cv2.putText(self.image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Classificacao YOLO", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""
# --- EXEMPLO DE USO ---

# Exemplo Detecção (usando a nova estrutura de objetos)
# Certifique-se que o caminho do modelo e imagem existem
yolo = YOLODetection("runs/detect/yolov8n_platform_detector10/weights/best.onnx", "image10.png", (640, 640), 0.5, 0.45)

# Agora você recebe uma lista de objetos
detections = yolo.get_detections()

for box in detections:
    print(f"Objeto encontrado: {box}")  # Usa o __repr__ da classe DetectionBox
    # Você pode acessar atributos: box.x, box.score, etc.

yolo.imgshow()

sla = YOLOClass("runs/classify/yolo_manometer_legivel/weights/best.onnx", "manometroilegivel.jpg", (640, 640), ['ilegivel', 'legivel'])
print(sla.classresult())
sla.imgshow()
"""
