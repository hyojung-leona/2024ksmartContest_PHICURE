#from ultralytics import YOLO
#model = YOLO('best.pt')
#model.predict(source=0, imgsz=640, conf=0.6, show=0)


import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('best.pt')  

# 웹캠 초기화
cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 예측
    results = model(frame, imgsz=640, conf=0.6)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 출력 프레임 보여주기
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

