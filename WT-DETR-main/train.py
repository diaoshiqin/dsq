import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    model = RTDETR('D:/Model/WT-DETR/WT-DETR-main/ultralytics/cfg/models/WT-DETR/wt-detr.yaml')
    model.load('D:/Model/WT-DETR/WT-DETR-main/wtdetr-r50.pt')
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=1,
                workers=0,
                device='0',
                # resume='runs/train/exp/weights/last.pt', # last.pt path
                project='runs/train',
                name='exp',
                )