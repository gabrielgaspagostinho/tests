import cv2
from ultralytics import YOLO
import math
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("runs/detect/yolo_manometer_detector5/weights/last.pt")
yololegivel = YOLO("runs/classify/yolo_manometer_legivel/weights/best.pt")
yoloclass = YOLO("runs/classify/yolo_manometer_class3/weights/last.pt")
# object classes
classNames = ["manometer"]
iflegivel = 0

while True:
    success, img = cap.read()
    results = model.track(img, conf=0.7)

    annotated_frame = results[0].plot()
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            croped = img[y1:y2, x1:x2]
            legivel = yololegivel.predict(croped)
            iflegivel = int(legivel[0].probs.top1)
            verde = (0, 255, 0)
            vermelho = (255, 0, 0)
            # put box in cam

            print(iflegivel)
            if iflegivel == 1:
                classe = yoloclass.predict(croped)
                color = verde
            if iflegivel == 2:
                color = vermelho
            if iflegivel == 0:
                color = (255, 0, 255)

            print(iflegivel)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            text = classNames[cls]
            thickness = 2
            print(x1, y1, x2, y2)
            valmano = ""
            if iflegivel == 1:
                print(classe[0].names)
                print(classe[0].probs.top1)
                valmano = str(classe[0].names[int(classe[0].probs.top1)])
            cv2.putText(img, text, org, font, fontScale, color, thickness)
            cv2.putText(img, valmano, [x1+5, y2-5], font, 0.8, color, thickness)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()