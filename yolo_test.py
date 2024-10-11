from ultralytics import YOLO
import numpy as np
# 初始化 YOLO 模型
model = YOLO("yolo11n.pt")  # 確保你有最新的 YOLO 模型權重文件
unique_person_ids = set()  # 用於存儲唯一的追蹤 ID

# 開始進行追蹤
results = model.track(source="./example01.mp4", show=True, stream=True)  # 使用 stream=True 即時顯示影像

# 迭代每一幀並處理追蹤結果
for result in results:
    frame = result.orig_img  # 獲取當前幀的影像
    for track in result.boxes:
        # 只處理偵測到的人物
        if track.cls == 0:  # 假設類別 0 是 "person" 類別
            person_id = track.id
            if person_id is not None:
                unique_person_ids.add(person_id)
