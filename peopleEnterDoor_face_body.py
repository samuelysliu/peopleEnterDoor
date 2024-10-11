import cv2
import numpy as np

video_path = './example01.mp4'

# 讀取影像，並且讓影像變成偵做後續分析
cap = cv2.VideoCapture(video_path)

# 加載 DNN 模型 用於面部辨識
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# 加載 YOLOv4 模型進行人體檢測（需要 YOLO 的模型文件）
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

frame_interval = 5  # 每x偵只分析一次，避免負載量過高

# Counter for people boarding
boarding_count = 0

while True:
    ret, frame = cap.read()
    #ret true of false to dectect video is open correct or not
    if not ret:
        break
    #取得現在偵的畫面，以利後續針對圖片作分析
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if current_frame % frame_interval == 0:
        #影片截圖視窗的尺寸
        resized_frame = cv2.resize(frame, (640, 480))

       # 人臉檢測
       #blobFromImage(圖片大小, 放大比例, 將分析的影像尺寸縮放, 將影像RGB消除光線顏色, 是否交換紅色藍色, 是否裁減影像)
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()

        # 繪製人臉框
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # 調整信心度閾值
                box = detections[0, 0, i, 3:7] * np.array([resized_frame.shape[1], resized_frame.shape[0], resized_frame.shape[1], resized_frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(resized_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 人體檢測
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:  # 只檢測人類（class_id == 0）
                    center_x = int(detection[0] * resized_frame.shape[1])
                    center_y = int(detection[1] * resized_frame.shape[0])
                    w = int(detection[2] * resized_frame.shape[1])
                    h = int(detection[3] * resized_frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 顯示結果
        cv2.imshow('Frame with Face and Body Detections', resized_frame)

        # Wait for user to press a key to move to the next frame or quit
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit the loop early
            break

cap.release()
cv2.destroyAllWindows()
print(boarding_count)
