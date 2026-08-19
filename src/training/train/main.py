import os
import mlflow
from ultralytics import YOLO, settings


# 1.model configuration

MODEL_NAME = "yolo26s"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. paths config

PRETEINED_MODEL_PATH = os.path.join(CURRENT_DIR, "pretrained", f"{MODEL_NAME}.pt")
DATA_CONFIG_PATH = os.path.join(
    CURRENT_DIR,
    "../../../../../datasets/Clear_v1.v2-clear_v1.1_-3-tags.yolo26/data.yaml",
)
LOCAL_RUNS_DIR = os.path.join(CURRENT_DIR, "runs", "YOLO_Detection")

# 3. mlflow config
MLFLOW_DB_PATH = os.path.join(CURRENT_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

# 4. ultralytics settings
settings.update({"mlflow": True, "runs_dir": LOCAL_RUNS_DIR})


# 5. training configuration
def train_detection():
    print(f"Starting YOLO26s detection training by {MODEL_NAME} ...")

    # load model (GPU)
    model = YOLO(PRETEINED_MODEL_PATH).to("cuda")

    # mlflow configuration
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "ClearSky"
    os.environ["MLFLOW_RUN_NAME"] = f"{MODEL_NAME}_detect"

    # train model
    model.train(
        data=DATA_CONFIG_PATH,
        epochs=100,
        batch=4,  # RTX 3050 (4 GB VRAM): evita OOM en TaskAlignedAssigner
        imgsz=640,
        patience=20,  # curva ruidosa con dataset chico; 10 cortaba muy pronto
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,  # cosine decay para pulir la convergencia
        workers=4,  # 8 dataloaders saturaban el CPU de la laptop
        cache=True,  # dataset ~85 MB: cabe en RAM, dataloader mucho más rápido
        task="detect",
        project=LOCAL_RUNS_DIR,
        name=f"{MODEL_NAME}_detect",
        exist_ok=True,
        save=True,
    )

    print(f"Training completed. Model saved in {LOCAL_RUNS_DIR}/{MODEL_NAME}_detect")


if __name__ == "__main__":
    train_detection()
