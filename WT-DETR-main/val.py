import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('D:/Model/RTDETR-v3/RTDETR-main/runs/train/rtdetr-r50.New-WaveletPool-bn-relu+7WTconv-1280/weights/best.pt')
    model.val(data='dataset/VisDrone.yaml',
              split='test',
              imgsz=1280,
              batch=2,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
