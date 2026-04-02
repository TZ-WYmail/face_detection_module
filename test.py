from ultralytics import YOLO

# 加载预训练 YOLO11n 模型
model = YOLO("yolo11n.pt")

# 对图片进行预测并保存结果
model.predict(
source="https://ultralytics.com/images/bus.jpg",
save=True,
imgsz=320,
conf=0.5,
show=True
)