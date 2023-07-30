import itertools
import random
import cv2
from pathlib import Path
import torch
from time import sleep
import numpy as np
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm


class LoadDataset(object):
    FORMAT = ['.mp4']
    mean = np.array([0.389, 0.379, 0.341])
    std = np.array([0.279, 0.273, 0.275])

    def __init__(self, video_dir, cam_idx, outputShape=(182, 182), numFrame=4):
        if cam_idx is None:
            src = Path(video_dir) if isinstance(video_dir, str) else video_dir
            assert src.exists(), f'{src.as_posix()} not found'
            assert src.suffix in self.FORMAT, f'{src.suffix} does not support'
            src = src.as_posix()
        else:
            assert isinstance(cam_idx, int), f"webcam must be an integer"
            src = cam_idx
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            self.frames = -1
        else:
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.outputShape = outputShape if len(outputShape) == 2 else (outputShape, outputShape)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.numFrame = numFrame
        assert self.cap.isOpened(), f'open the resource failed'

    def __len__(self):
        return self.frames

    def __iter__(self):
        return self

    def get_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        sleep(1 / self.fps)
        re_size_frame = frame.copy()
        re_size_frame = cv2.resize(re_size_frame, self.outputShape)
        re_size_frame = cv2.cvtColor(re_size_frame, cv2.COLOR_BGR2RGB)
        re_size_frame = re_size_frame.astype(np.float32)
        re_size_frame /= 255.
        re_size_frame -= self.mean
        re_size_frame /= self.std
        return frame, re_size_frame

    def numpy2tensor(self, frame):
        frame = cv2.resize(frame, self.outputShape)  # HWC
        tensorFrame = torch.from_numpy(frame)
        tensorFrame = torch.unsqueeze(tensorFrame, 0)  # 0HWC
        return tensorFrame

    def tensor_normalize(self, tensor, mean, std, func=None):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        if func is not None:
            tensor = func(tensor)
        tensor = tensor - mean
        tensor = tensor / std
        return tensor


class LoadSampleforVideoClassify(Dataset):
    VID_FORMATS = ['.asf', '.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv',
                   '.gif']  # acceptable video suffixes
    mean = (0., 0., 0.)
    std = (1., 1., 1.)
    use_BGR = False

    def __init__(self, root, augment=True, cache=True, prefix="", backend='pyav'):
        self.transform = None
        self.root = root = Path(root) if isinstance(root, str) else root

        self.prefix = prefix
        from torchvision.datasets.folder import make_dataset
        classes = [x.name for x in root.iterdir() if x.is_dir()]
        classes.sort()
        class_to_indx = {class_name: i for i, class_name in enumerate(classes)}
        self.classes = classes
        self.class_to_indx = class_to_indx
        assert len(class_to_indx) > 0, f"dataset not found ({root.as_posix()})"
        self.samples = make_dataset(directory=root.as_posix(),
                                    class_to_idx=class_to_indx,
                                    extensions=tuple(self.VID_FORMATS))
        self.imgsz = 224
        self.clip_len = 16
        self.step = 5
        self.pin_memory = False
        backend = backend.lower()
        try:
            if backend == "pyav":
                torchvision.set_video_backend(backend=backend)
        except Exception as ex:
            torchvision.set_video_backend("pyav")
            print(f"{self.prefix}fallback to PyAV backend \n{ex}")

    def prepare(self):
        """prepare dataset"""
        if sum(self.mean) == 0 and sum(self.std) == 1:
            self.calculateMeanStd()
        else:
            print(f"{self.prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")
        self.step = max(1, int(self.step))
        self.clip_len = max(1, int(self.clip_len))

        print(
            f"{self.prefix}total {len(self.samples)} samples with {len(self.classes)} classes, "
            f"frame length: {self.clip_len}, step frame: {self.step}")
        self.transform = transforms.Compose([
            transforms.Lambda(lambd=lambda x: self.randomDropChannel(x, 0.1)),
            transforms.Lambda(lambd=lambda x: self.randomDropFrame(x, 0.1)),
            transforms.Lambda(lambd=lambda x: self.hflip(x, 0.1)),
            transforms.Lambda(lambd=lambda x: self.vflip(x, 0.1)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.Lambda(lambd=lambda x: self.rgb_2_gray(x, 0.1)),
            transforms.Lambda(lambd=lambda x: self.normalize(x, self.mean, self.std))
        ])

    def dataset_analysis(self):
        """for now only return number of frame per classes"""
        dict_ = {target: 0 for sample, target in self.samples}
        for x, i in self.samples:
            vid = torchvision.io.VideoReader(x, "video")
            metadata = vid.get_metadata()
            total_frames = metadata["video"]['duration'][0] * metadata["video"]['fps'][0]
            dict_[i] += total_frames
        return dict_, self.classes

    def calculateMeanStd(self):
        """Calculate mean, std. https://kozodoi.me/blog/20210308/compute-image-stats"""
        samples = self.samples
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        pbar = tqdm(samples, total=len(samples))
        transform = transforms.Compose([
            transforms.Lambda(lambd=lambda x: self.normalize(x, mean=self.mean, std=self.std))])

        for i, (path, _) in enumerate(pbar):
            video, _ = self.loadSample(path, transform=transform, dtype=torch.float32)
            psum += video.sum(axis=[0, 2, 3])
            psum_sq += (video ** 2).sum(axis=[0, 2, 3])
            pbar.set_description(f"{self.prefix}Collecting data to calculate mean, std...")
            if i >= len(samples) - 10:
                count = len(samples) * self.clip_len * self.imgsz * self.imgsz
                total_mean = psum / count
                total_var = (psum_sq / count) - (total_mean ** 2)
                total_std = torch.sqrt(total_var)
                self.mean = total_mean.cpu().numpy().tolist()
                self.std = total_std.cpu().numpy().tolist()
                pbar.set_description(f"{self.prefix}Calculating...")
                pbar.set_description(f"{self.prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        video = self.loadSample(path=path, transform=self.transform, dtype=torch.float32)[0]
        video = torch.permute(video, dims=[1, 0, 2, 3])  # NCHW -> CNHW
        return video, target

    def loadSample(self, path=None, transform=None, dtype=torch.uint8):
        """choice a random sample to plot"""
        target = 0
        resize_transform = transforms.Compose([transforms.Resize((self.imgsz, self.imgsz), antialias=False)])
        if path is None:
            path, target = random.choice(self.samples)
        vid = torchvision.io.VideoReader(path, "video")
        vid.set_current_stream("video")
        metadata = vid.get_metadata()
        # Seek and return frames
        n_length = self.clip_len * self.step
        fps = metadata["video"]['fps']
        fps = fps[0] if isinstance(fps, list) else fps
        max_seek = metadata["video"]['duration'][0] - (n_length / fps)
        start = random.uniform(0., max_seek)
        video = torch.zeros([self.clip_len, 3, self.imgsz, self.imgsz], dtype=dtype)
        for i, frame in enumerate(itertools.islice(vid.seek(start, keyframes_only=True), 0, n_length, self.step)):
            video[i, ...] = resize_transform(frame['data'])

        if transform:
            video = transform(video)
        return video, self.classes[target]

    @staticmethod
    def collect_fn(batch):
        videos, target = zip(*batch)
        return torch.stack(videos, 0), torch.tensor(target, dtype=torch.long)

    @staticmethod
    def rgb_2_gray(inputs: torch.Tensor, p=0.5) -> torch.Tensor:
        """convert multiple rgb image to gray"""
        gray = inputs.float()
        n_shape = inputs.dim()
        if random.random() <= p:
            if n_shape == 4:
                for i, rgb in enumerate(gray):
                    r, g, b = rgb[0, ...], rgb[1, ...], rgb[2, ...]
                    g = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    gray[i, ...] = torch.stack([g, g, g])
            elif n_shape == 3:
                r, g, b = gray[0, ...], gray[1, ...], gray[2, ...]
                g = 0.2989 * r + 0.5870 * g + 0.1140 * b
                gray = torch.stack([g, g, g])
            else:
                raise f"{n_shape} dimension does not support."
        return gray.round().to(torch.uint8)

    @staticmethod
    def randomDropFrame(inputs: torch.Tensor, p=0.5) -> torch.Tensor:
        if random.random() <= p:
            n_dims = inputs.dim()
            if n_dims == 4:
                frame_idx = random.randint(0, inputs.shape[0] - 1)
                inputs[frame_idx, ...] = 0
            else:
                inputs[...] = 0
        return inputs

    @staticmethod
    def randomDropChannel(inputs: torch.Tensor, p=0.5) -> torch.Tensor:
        if random.random() <= p:
            channel = random.randint(0, 2)
            n_dims = inputs.dim()
            if n_dims == 4:
                inputs[:, channel, ...] = 0
            else:
                inputs[channel, ...] = 0

        return inputs

    @staticmethod
    def vflip(inputs: torch.Tensor, p=0.5) -> torch.Tensor:
        n_dims = len(inputs.shape)
        if random.random() <= p:
            if n_dims == 4:
                inputs = torch.flip(inputs, dims=[2])
            else:
                inputs = torch.flip(inputs, dims=[1])
        return inputs

    @staticmethod
    def hflip(inputs: torch.Tensor, p=0.5) -> torch.Tensor:
        n_dims = inputs.dim()
        if random.random() <= p:
            if n_dims == 4:
                inputs = torch.flip(inputs, dims=[3])
            else:
                inputs = torch.flip(inputs, dims=[2])
        return inputs

    @staticmethod
    def normalize(inputs: torch.Tensor, mean: tuple, std: tuple, pixel_max_value=255.) -> torch.FloatTensor:
        """input float tensor in NCHW or CHW format and return the same format"""
        n_dims = inputs.dim()
        inputs = inputs.float()
        inputs /= pixel_max_value
        if n_dims == 4:
            n_dept, c, h, w = inputs.shape
            assert c == len(mean), f"len of mean ({len(mean)}) must be equal to image channel ({c})"
            for n in range(n_dept):
                for x in range(c):
                    inputs[n, x, ...] = (inputs[n, x, ...] - mean[x]) / (std[x])
        elif n_dims == 3:
            c, h, w = inputs.shape
            assert c == len(mean), f"len of mean ({len(mean)}) must be equal to image channel ({c})"
            for x in range(c):
                inputs[x, ...] = (inputs[x, ...] - mean[x]) / (std[x])
        else:
            raise f"inputs tensor must be 3 or 4 dimension, got {n_dims}"
        return inputs
