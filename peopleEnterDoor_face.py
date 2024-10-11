import cv2
import numpy as np

video_path = './example01.mp4'

# Read video and covert to frame
cap = cv2.VideoCapture(video_path)

# 加載 DNN 模型 用於面部辨識
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

frame_interval = 5  # 每x偵只分析一次，避免負載量過高

# 定義畫面須辨識範圍
roi_top = 40  # Y-coordinate of top of ROI
roi_bottom = 420  # Y-coordinate of bottom of ROI
roi_left = 100  # X-coordinate of left of ROI
roi_right = 550  # X-coordinate of right of ROI

# Counter for people boarding
boarding_count = 0

# Analyze the video using HOG person detection, focusing on the ROI for boarding
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if current_frame % frame_interval == 0:
        #影片截圖視窗的尺寸
        resized_frame = cv2.resize(frame, (640, 480))
        roi_frame = resized_frame[roi_top:roi_bottom, roi_left:roi_right] #roi_top 代表Y的範圍，roi_left 代表水平範圍

        # 準備 DNN 輸入資料，將圖像大小調整為 300x300 並進行均值減去
        blob = cv2.dnn.blobFromImage(roi_frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        # 進行人臉檢測
        net.setInput(blob)
        detections = net.forward()
        
        # Draw the ROI on the frame with a red rectangle
        cv2.rectangle(resized_frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

        # 繪製檢測到的人臉框
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 只保留信心度大於50%的檢測結果
                box = detections[0, 0, i, 3:7] * np.array([roi_frame.shape[1], roi_frame.shape[0], roi_frame.shape[1], roi_frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(resized_frame, (startX + roi_left, startY + roi_top), (endX + roi_left, endY + roi_top), (0, 255, 0), 2)
                boarding_count += 1
                # Display the frame
                cv2.imshow('Frame with ROI and Detections', resized_frame)
        

        # Wait for user to press a key to move to the next frame or quit
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit the loop early
            break

cap.release()
cv2.destroyAllWindows()
print(boarding_count)
