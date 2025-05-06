from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLO11x模型
    model = YOLO("../ly_js/weights/yolo11n.pt")

    # SKU-110数据集进行训练
    train_results = model.train(
        data="SKU-110K.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size for training
        device='0',
        batch=16
    )