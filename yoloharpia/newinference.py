import cv2
import numpy as np


# pre processamento da imagem para redimensionar ela para o tamanho padrao da inferencia e depois voltar
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Redimensiona a imagem para o tamanho de entrada da rede YOLO mantendo a proporção.
    Adiciona bordas (padding) se necessário para evitar distorção do objeto.
    """
    shape = im.shape[:2]  # Formato atual [altura, largura]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calcula o fator de escala (ratio) para caber no novo formato
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Não aumenta a imagem, apenas reduz se for maior que o alvo
        r = min(r, 1.0)

    # Novo tamanho após redimensionamento, mas antes do preenchimento das bordas
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Calcula o tamanho das bordas necessárias (padding)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # Ajusta o preenchimento para ser múltiplo de 64
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:  # Ignora a proporção e estica (não recomendado)
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2  # Divide por 2 para centralizar a imagem no quadro
    dh /= 2

    # Executa o redimensionamento físico
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Adiciona a moldura colorida (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, (r, r) if not isinstance(r, tuple) else r, (dw, dh)



# classe de cada box de deteccao
class DetectionBox:

    def __init__(self, x, y, w, h, score, class_id):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.score = float(score)  # Nível de confiança (0 a 1)
        self.class_id = int(class_id)  # Índice da categoria detectada

    def __repr__(self):
        return f"Box(ID: {self.class_id}, Conf: {self.score:.2f}, Pos: [{self.x},{self.y},{self.w},{self.h}])"


#classe base para as inferencias
class YOLOBase:
    #Classe Pai: Gerencia o carregamento do modelo e o preparo da imagem.

    def __init__(self, model_path, input_size=(640, 640)):
        self.input_size = input_size
        self.net = self._load_model(model_path)
        self.image = None  # Imagem original (matriz OpenCV)
        self.ratio = (1.0, 1.0)  # Fator de escala do letterbox
        self.dw_dh = (0, 0)  # Deslocamento das bordas (padding)

    def _load_model(self, path):
        #Carrega o arquivo ONNX usando o módulo DNN do OpenCV
        print(f"--- Carregando modelo ONNX: {path} ---")
        return cv2.dnn.readNetFromONNX(path)

    def set_image(self, image_input):
        # define a imagem para processar e disdingue se é imagem do openCv ou um path
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            self.image = image_input.copy()  # mantém uma cópia local segura

        if self.image is None:
            print("ERRO: Fonte de imagem inválida!")

    def _preprocess(self):
        #Prepara a imagem para a rede: Letterbox + Normalização + Blob
        if self.image is None: return None

        # 1. Ajusta tamanho e bordas
        img_padded, self.ratio, self.dw_dh = letterbox(self.image, new_shape=self.input_size, auto=False)

        # 2. Converte para 'Blob': Normaliza para 0-1, inverte cores (BGR para RGB)
        blob = cv2.dnn.blobFromImage(img_padded, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        return blob


class YOLODetection(YOLOBase):
    # detecção de boxes

    def run_inference(self, score_threshold=0.5, nms_threshold=0.45):
        blob = self._preprocess()
        if blob is None: return []

        self.net.setInput(blob)
        outputs = self.net.forward()  # executa a rede neural

        # o yolo retorna os dados com formato (1, 84, 8400), transpomos para processar
        outputs = np.array([cv2.transpose(outputs[0])])

        temp_boxes, temp_scores, temp_class_ids = [], [], []
        dw, dh = self.dw_dh
        rw, rh = self.ratio

        for row in outputs[0]:
            # as primeiras 4 colunas são x, y, w, h. As restantes são scores das classes
            classes_scores = row[4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)

            if max_score >= score_threshold:
                cx, cy, w, h = row[0:4]

                # REVERSÃO DO LETTERBOX: Volta as coordenadas para a escala da imagem original
                left = int((cx - w / 2 - dw) / rw)
                top = int((cy - h / 2 - dh) / rh)
                width, height = int(w / rw), int(h / rh)

                temp_boxes.append([left, top, width, height])
                temp_scores.append(float(max_score))
                temp_class_ids.append(max_class_loc[1])

        # NMS (Non-Maximum Suppression): elimina detecções sobrepostas do mesmo objeto
        indices = cv2.dnn.NMSBoxes(temp_boxes, temp_scores, score_threshold, nms_threshold)

        # faz uma lista de objetos com as boxes com a filtragem do NMS
        self.detections = [
            DetectionBox(*temp_boxes[i], temp_scores[i], temp_class_ids[i])
            for i in (indices.flatten() if len(indices) > 0 else [])
        ]
        return self.detections

    def imgshow(self, window_name="Detecção"):
        if self.image is None: return
        canvas = self.image.copy()
        for box in self.detections:
            cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2)
            label = f"ID {box.class_id}: {box.score:.2f}"
            cv2.putText(canvas, label, (box.x + 5, box.y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 2)

        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)


class YOLOClass(YOLOBase):
    # inferencia de classificação

    def run_inference(self, class_names):
        blob = self._preprocess()
        if blob is None: return None

        self.net.setInput(blob)
        outputs = self.net.forward()

        # Softmax: Transforma os resultados brutos em probabilidades de 0 a 100%
        scores = outputs[0]
        probs = np.exp(scores - np.max(scores))
        probs /= probs.sum()

        class_id = np.argmax(probs)
        self.result = {
            "id": int(class_id),
            "score": float(probs[class_id]),
            "name": class_names[class_id] if class_id < len(class_names) else str(class_id)
        }
        return self.result

    def imgshow(self, window_name="Classificação"):
        if self.image is None or not hasattr(self, 'result'): return
        canvas = self.image.copy()
        text = f"{self.result['name']} ({self.result['score']:.2f})"
        cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)



# debug do detection
det = YOLODetection("runs/detect/yolov8n_platform_detector10/weights/best.onnx")
det.set_image("image10.png")
lista_objetos = det.run_inference(score_threshold=0.5)
print(f"Objetos encontrados: {lista_objetos}")
det.imgshow()

# debug do class
cls = YOLOClass("runs/classify/yolo_manometer_legivel/weights/best.onnx")
cls.set_image("fotomanometrofds.jpg")
resultado = cls.run_inference(class_names=['ilegivel', 'legivel'])
print(f"Classificação final: {resultado}")
cls.imgshow()

cv2.destroyAllWindows()
