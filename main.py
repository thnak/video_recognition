import argparse
from utils.dataloader import LoadDataset
from utils.model import Model
from utils.plots import plot
from pathlib import Path
import cv2
import numpy as np
import json
import torch
from time import time


def run(**kwargs):
    weight = Path(kwargs["weight"])
    src = Path(kwargs["src"])
    webcam = kwargs["webcam"]
    if webcam <= -1:
        webcam = None

    assert weight.exists(), f"weight not found, your {weight.as_posix()}"
    assert src.exists(), f"src not found, your {src.as_posix()}"
    model = Model(weight.as_posix())

    kineDir = Path(kwargs["json_classes"])
    if kineDir.exists():
        with open(kineDir.as_posix(), "r") as f:
            kine_json = json.load(f)
        id_to_classname = {}

        for k, v in kine_json.items():
            id_to_classname[v] = str(k).replace('"', "")
    else:
        id_to_classname = model.classes
    b, c, n, h, w = model.imgsz

    data = LoadDataset(src.as_posix(),
                       cam_idx=webcam, outputShape=(w, h), numFrame=n)
    data.mean = model.mean
    data.std = model.std

    frames = []
    label = "None"
    sample_rate = model.sampling_rate
    for i, (frame0, frame1) in enumerate(data):
        if i % sample_rate == 0:
            frames.append(frame1)
        if len(frames) == n:
            for ix in range(len(frames)):
                frames[ix] = np.expand_dims(frames[ix], 0)  # 1HWC
            frames = np.concatenate(frames, 0)  # 4HWC
            frames = np.expand_dims(frames, 0)  # #14HWC
            frames = np.transpose(frames, [0, 4, 1, 2, 3])  # BCNHW
            t0 = time()
            pred = model(frames)
            t1 = time() - t0
            frames = []
            val, inx = torch.topk(pred, dim=-1, k=1)
            val = val.detach().numpy().round(3)
            label = f"{id_to_classname[int(inx)]} {val}"
        frame0 = plot(frame0, label)
        cv2.namedWindow("a", cv2.WINDOW_NORMAL)
        cv2.imshow("a", frame0)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video regnition with 3D CNN")
    parser.add_argument("--weight", default="", type=str, help="model weight")
    parser.add_argument("--src", default="", type=str, help="video source directory")
    parser.add_argument("--json_classes", default="class_name.json", type=str, help="json name map (optional)")
    parser.add_argument("--webcam", default=-1, type=int, help="use webcam (idx>=0)")
    opt = parser.parse_args()

    if all([len(opt.weight), len(opt.src)]):
        run(**opt.__dict__)
