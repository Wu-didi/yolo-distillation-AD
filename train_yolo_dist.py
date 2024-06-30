from models.yolo import Model, parse_model
import argparse
from utils.general import check_yaml, print_args, set_logging
from utils.torch_utils import select_device
from pathlib import Path
import torch

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s_hefei_test.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    # print_args(FILE.stem, opt)
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    y = model(img, profile=True)
    feature = y[-1]
    print(len(feature))
    print("extract_feature shape: ", feature[0].shape)
    