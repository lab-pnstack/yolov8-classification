from ultralytics import settings
from ultralytics import YOLO
import torch

# set env GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

is_preload = True
def run():
    torch.multiprocessing.freeze_support()
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    results = model.train(data='/mnt/d/support/tam.bn/filterdataa/datasets', epochs=15, imgsz=320)

if __name__ == '__main__':
    run()