import json

import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch


class Model:
    def __init__(self, modelDir):
        modelDir = Path(modelDir) if isinstance(modelDir, str) else modelDir
        self.modelType = ""
        if modelDir.suffix == ".onnx":
            print(f"Start with onnx model")
            self.modelType = "onnx"
            providers = ort.get_available_providers()
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opt.enable_mem_pattern = False if 'DmlExecutionProvider' in providers else True
            sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_opt.use_deterministic_compute = True
            self.runer = ort.InferenceSession(modelDir.as_posix(), sess_opt, providers=providers)
            self.runer.enable_fallback()
            meta = self.runer.get_modelmeta().custom_metadata_map
            self.classes = meta.get("names", None)
            self.mean = eval(meta.get("mean", "[0, 0, 0]"))
            self.std = eval(meta.get("std", "[1, 1, 1]"))
            self.sampling_rate = int(meta.get("sampling_rate", "5"))
            if self.classes:
                self.classes = eval(self.classes)
            self.imgsz = self.runer.get_inputs()[0].shape
            self.input_names = [x.name for x in self.runer.get_inputs()]
            self.output_names = [x.name for x in self.runer.get_outputs()]
        else:
            self.modelType = "torch"
            print(f"Start with torchScript model")
            extra_files = {'config.txt': ''}
            self.runer = torch.jit.load(modelDir.as_posix(), _extra_files=extra_files)
            meta = extra_files["config.txt"]
            meta = json.loads(meta)
            self.mean = meta['mean']
            self.std = meta['std']
            self.imgsz = meta['input_shape']
            self.classes = meta["names"]
            self.sampling_rate = meta['sampling_rate']

    def __call__(self, inputs) -> torch.Tensor:
        if self.modelType == "onnx":
            feeds = {}
            for x in self.input_names:
                feeds[x] = inputs
            pred = self.runer.run(output_names=self.output_names, input_feed=feeds)[0]
            return torch.from_numpy(pred)[0]
        else:
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            return self.runer(inputs)
