from ultralytics import YOLO

def train(pretrain='../weights/yolo11n.pt', yaml="js-multi-act.yaml"):
    # 加载YOLO11模型
    model = YOLO(pretrain)

    # js数据集进行训练
    train_results = model.train(
        data=yaml,  # Path to dataset configuration file
        epochs=200,  # Number of training epochs
        imgsz=640,  # Image size for training
        device='0',
        batch=16,
        amp=True
    )


if __name__ == '__main__':
    pretrain = 'weights/last.pt'
    yaml = 'js-multi-act.yaml'
    train(pretrain=pretrain, yaml=yaml)