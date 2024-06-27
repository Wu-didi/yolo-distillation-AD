# python train.py --device 6,7 --batch 256 --cfg yolov5s-p2-v2.yaml
# python train.py --device 6,7 --batch 128 --cfg yolov5s_hefei_v2.4.yaml
# python train.py --device 2,3 --batch 64 --cfg yolov5m-p2.yaml
# yolov5s-p2-hefei-v2.4.yaml
#  python train.py --cfg yolov5s_hefei_v2.4_CA.yaml  --batch 32 --imgsz 640
python val.py --data /home/wudi/code/yolov5/data/hefei_v2.3.yaml \
--weights /home/wudi/code/yolov5/runs/train/hefeiv2.3-exp7/weights/best.pt --img 640