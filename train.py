import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=' ',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=10,
                workers=4,
                device='',
                optimizer='AdamW', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )