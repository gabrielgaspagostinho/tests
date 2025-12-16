# Guia de Integração: Módulo de Inferência YOLO (ONNX)

Este documento descreve como importar e utilizar o script `inference_harpia.py` como um módulo em seus próprios projetos Python.

O módulo permite realizar inferências de **Detecção de Objetos** e **Classificação** utilizando modelos YOLOv8 exportados para ONNX, sem depender da biblioteca `ultralytics`.

## 📦 Pré-requisitos

Certifique-se de que o arquivo `inference_harpia.py` esteja no mesmo diretório do seu script principal (ou no `PYTHONPATH`).

Dependências necessárias:
```bash
pip install opencv-python numpy
```

## 🚀 Como Importar

No seu script Python (ex: `main.py` ou `app_robot.py`), importe as classes principais:

```python
# Importa as classes de inferência
from inference_harpia import YOLODetection, YOLOClass
```

---

## 🔍 Detecção de Objetos (`YOLODetection`)

Use esta classe para identificar objetos, desenhar caixas e obter coordenadas.

### 1. Inicialização

```python
# Caminho do modelo e da imagem
model_path = "models/best_detect.onnx"
img_path = "data/input.jpg"

# Instancia o detector
detector = YOLODetection(
    MODEL_PATH=model_path,
    IMAGE_PATH=img_path,
    INPUT_SIZE=(640, 640),  # Tamanho de entrada do modelo (geralmente 640)
    SCORE_THRESHOLD=0.5,    # Confiança mínima (0.0 a 1.0)
    NMS_THRESHOLD=0.45      # Limiar para remover caixas sobrepostas
)
```

### 2. Obtendo Resultados (`DetectionBox`)

O método `get_detections()` retorna uma lista de objetos do tipo `DetectionBox`. Cada objeto possui atributos diretos, facilitando o acesso aos dados.

```python
# Obtém a lista de objetos detectados
objetos = detector.get_detections()

# Itera sobre os resultados
for obj in objetos:
    print(f"--- Objeto Detectado ---")
    print(f"Classe ID: {obj.class_id}")
    print(f"Score: {obj.score:.2f}")
    
    # Acessando coordenadas e dimensões
    print(f"Posição X: {obj.x}, Y: {obj.y}")
    print(f"Largura: {obj.w}, Altura: {obj.h}")
    
    # Exemplo de lógica de decisão
    if obj.class_id == 0 and obj.score > 0.8:
        print(">> Alvo prioritário encontrado!")
```

### 3. Visualização

Para visualizar o resultado em uma janela do OpenCV:

```python
detector.imgshow()
```

---

## 🏷️ Classificação de Imagens (`YOLOClass`)

Use esta classe para classificar o conteúdo global de uma imagem (ex: "Dia" vs "Noite", "Defeito" vs "Normal").

### 1. Inicialização

Você deve fornecer a lista de nomes das classes **na mesma ordem** em que o modelo foi treinado.

```python
# Definição das classes
minhas_classes = ['ilegivel', 'legivel']

# Instancia o classificador
classificador = YOLOClass(
    MODEL_PATH="models/best_classify.onnx",
    IMAGE_PATH="data/manometro.jpg",
    INPUT_SIZE=(640, 640),  # Verifique se seu modelo CLS usa 224 ou 640
    CLASS_NAMES=minhas_classes
)
```

### 2. Obtendo Resultados

O método `classresult()` retorna um dicionário simples com os dados da predição vencedora.

```python
resultado = classificador.classresult()

# O retorno é um dicionário: {'id': int, 'score': float, 'name': str}
print(f"Resultado: {resultado['name']}")
print(f"Confiança: {resultado['score']}")
```

### 3. Visualização

```python
classificador.imgshow()
```

---

## ⚠️ Notas sobre Exportação do Modelo

Para garantir compatibilidade com o OpenCV, exporte seus modelos YOLO (`.pt`) para ONNX utilizando o argumento `opset=12`.

**Comando de exportação(em um terminal com YOLO):**
```bash
yolo export model=best.pt format=onnx opset=12
```
