import os
import sys
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

pth = '/'.join(sys.path[0].split('/')[:-2]) 
sys.path.insert(0, pth)

import torch
import argparse
import whisper
import numpy as np


from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks import *

import time

start = time.time()

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args()
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
    cur_model = 'Focal-T'
elif 'focal' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focall_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

output_file = "results.txt"

def process_images_in_folder(folder_path):
    with open(output_file, "w") as file:
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))], key=lambda x: int(x.split('_')[1].split('.')[0]) + int(x.split('_')[0]*60))

        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            image = {'image': Image.open(image_path), 'mask': None}
            #print(f"Processing {filename}...")

            task = ["Panoptic"]
            result = inference(image, task)
            
            # 結果をファイルに書き込む
           #file.write(f"Result tags for {filename}: {task}\n")
           # print(f"Result tags for {filename}: {result}")

# フォルダのパスを指定
folder_path = './demo/seem/images'
process_images_in_folder(folder_path)

elapsed_time = time.time() - start
print(f'time: {elapsed_time}')
