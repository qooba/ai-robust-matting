"""
Inference video: Extract matting on video.

"""

import cv2
import torch
import os
import io
import shutil
from collections import namedtuple

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image

from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment
import moviepy.editor as mp
from moviepy.video.fx.all import crop
from PIL import Image, ImageColor


# --------------- Arguments ---------------

Args = namedtuple("Args", [
    "model_type",
    "model_backbone",
    "model_backbone_scale",
    "model_checkpoint",
    "model_refine_mode",
    "model_refine_sample_pixels",
    "model_refine_threshold",
    "model_refine_kernel_size",
    "video_src",
    "video_bgr",
    "video_target_bgr",
    "video_resize",
    "device",
    "preprocess_alignment",
    "output_dir",
    "output_types",
    "output_format"])




# --------------- Utils ---------------


class VideoWriter:
    def __init__(self, path, frame_rate, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    def add_batch(self, frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)


class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path)

    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += frames.shape[0]

    def _add_batch(self, frames, index):
        frames = frames.cpu()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = to_pil_image(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))


# --------------- Main ---------------




class VideoMatteService:

    def matting(self, video_src: str,
                video_bgr: str,
                output_dir: str,
                video_target_bgr: str = None,
                device: str = 'cuda',
                output_types = ['com'] ):

        args = Args(
            model_type="mattingrefine",                         # 'mattingbase', 'mattingrefine'
            model_backbone="resnet50",                          # 'resnet101', 'resnet50', 'mobilenetv2'
            model_backbone_scale=0.25,                          # 0.25
            model_checkpoint="/app/model.pth",
            model_refine_mode="sampling",                       # 'full', 'sampling', 'thresholding'
            model_refine_sample_pixels=80000,                   # 80000
            model_refine_threshold=0.7,                         # 0.7
            model_refine_kernel_size=3,                         # 3
            video_src=video_src,
            video_bgr=video_bgr,
            video_target_bgr=video_target_bgr,                  # Path to video onto which to composite the output (default to flat green)
            video_resize=None,                                  # None
            device=device,                                      # 'cpu', 'cuda'
            preprocess_alignment=False,
            output_dir=output_dir,
            output_types=output_types,                          # ['com', 'pha', 'fgr', 'err', 'ref']
            output_format="video"                               # 'video', 'image_sequences'
        )

        assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
            'Only mattingbase and mattingrefine support err output'
        assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
            'Only mattingrefine support ref output'

        device = torch.device(args.device)

        # Load model
        if args.model_type == 'mattingbase':
            model = MattingBase(args.model_backbone)
        if args.model_type == 'mattingrefine':
            model = MattingRefine(
                args.model_backbone,
                args.model_backbone_scale,
                args.model_refine_mode,
                args.model_refine_sample_pixels,
                args.model_refine_threshold,
                args.model_refine_kernel_size)

        model = model.to(device).eval()
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)


        # Load video and background
        vid = VideoDataset(args.video_src)
        bgr = [Image.open(args.video_bgr).convert('RGB')]
        dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
            A.PairApply(T.Resize(args.video_resize[::-1]) if args.video_resize else nn.Identity()),
            HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
            A.PairApply(T.ToTensor())
        ]))
        if args.video_target_bgr:
            dataset = ZipDataset([dataset, VideoDataset(args.video_target_bgr, transforms=T.ToTensor())])

        # Create output directory
        if os.path.exists(args.output_dir):
            if input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
                shutil.rmtree(args.output_dir)
            else:
                exit()
        os.makedirs(args.output_dir)


        # Prepare writers
        if args.output_format == 'video':
            h = args.video_resize[1] if args.video_resize is not None else vid.height
            w = args.video_resize[0] if args.video_resize is not None else vid.width
            if 'com' in args.output_types:
                com_writer = VideoWriter(os.path.join(args.output_dir, 'com.mp4'), vid.frame_rate, w, h)
            if 'pha' in args.output_types:
                pha_writer = VideoWriter(os.path.join(args.output_dir, 'pha.mp4'), vid.frame_rate, w, h)
            if 'fgr' in args.output_types:
                fgr_writer = VideoWriter(os.path.join(args.output_dir, 'fgr.mp4'), vid.frame_rate, w, h)
            if 'err' in args.output_types:
                err_writer = VideoWriter(os.path.join(args.output_dir, 'err.mp4'), vid.frame_rate, w, h)
            if 'ref' in args.output_types:
                ref_writer = VideoWriter(os.path.join(args.output_dir, 'ref.mp4'), vid.frame_rate, w, h)
        else:
            if 'com' in args.output_types:
                com_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'com'), 'png')
            if 'pha' in args.output_types:
                pha_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'pha'), 'jpg')
            if 'fgr' in args.output_types:
                fgr_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'fgr'), 'jpg')
            if 'err' in args.output_types:
                err_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'err'), 'jpg')
            if 'ref' in args.output_types:
                ref_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'ref'), 'jpg')


        # Conversion loop
        with torch.no_grad():
            for input_batch in tqdm(DataLoader(dataset, batch_size=1, pin_memory=True)):
                if args.video_target_bgr:
                    (src, bgr), tgt_bgr = input_batch
                    tgt_bgr = tgt_bgr.to(device, non_blocking=True)
                else:
                    src, bgr = input_batch
                    tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)
                src = src.to(device, non_blocking=True)
                bgr = bgr.to(device, non_blocking=True)

                if args.model_type == 'mattingbase':
                    pha, fgr, err, _ = model(src, bgr)
                elif args.model_type == 'mattingrefine':
                    pha, fgr, _, _, err, ref = model(src, bgr)
                elif args.model_type == 'mattingbm':
                    pha, fgr = model(src, bgr)

                if 'com' in args.output_types:
                    if args.output_format == 'video':
                        # Output composite with green background
                        com = fgr * pha + tgt_bgr * (1 - pha)
                        com_writer.add_batch(com)
                    else:
                        # Output composite as rgba png images
                        com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                        com_writer.add_batch(com)
                if 'pha' in args.output_types:
                    pha_writer.add_batch(pha)
                if 'fgr' in args.output_types:
                    fgr_writer.add_batch(fgr)
                if 'err' in args.output_types:
                    err_writer.add_batch(F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False))
                if 'ref' in args.output_types:
                    ref_writer.add_batch(F.interpolate(ref, src.shape[2:], mode='nearest'))

class VideoService:

    def __init__(self, video_matte: VideoMatteService):
        self.video_matte = video_matte

    def _generate_target_color(self, tempdir, size, duration, target_color):
        #use image background
        video_target_bgr=f"{tempdir}/video_target_bgr.mp4"
        video_target_bgr_img=f"{tempdir}/video_target_bgr.png"
        ##clips = [mp.ImageClip("/app/tmp/bgr_white.png").set_duration(my_clip.duration)]
        Image.new('RGB', size, color = ImageColor.getrgb(target_color)).save(video_target_bgr_img)
        clips = [mp.ImageClip(video_target_bgr_img).set_duration(duration)]

        concat_clip = mp.concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_target_bgr, fps=24)
        return video_target_bgr

    def _generate_target_image(self, tempdir, size, duration, target_image):
        video_target_bgr=f"{tempdir}/video_target_bgr.mp4"
        video_target_bgr_img=f"{tempdir}/video_target_bgr.png"

        im = Image.open(target_image)
        width, height = im.size

        left = (width - size[0])/2
        top = (height - size[1])/2
        right = (width + size[0])/2
        bottom = (height + size[1])/2

        im.crop((left, top, right, bottom)).save(video_target_bgr_img)
        clips = [mp.ImageClip(video_target_bgr_img).set_duration(duration)]

        concat_clip = mp.concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_target_bgr, fps=24)
        return video_target_bgr

    def _generate_target_video(self, tempdir, size, duration, target_video):
        clip = mp.VideoFileClip(target_video)
        video_target_bgr=f"{tempdir}/video_target_bgr.mp4"

        width, height = clip.size

        left = (width - size[0])/2
        top = (height - size[1])/2
        right = (width + size[0])/2
        bottom = (height + size[1])/2

        crop(clip, x1=left, y1=top, x2=right, y2=bottom).write_videofile(video_target_bgr)
        return video_target_bgr

    def process(self,
                tempdir: str,
                file_src: str,
                file_bgr: str,
                target_type: str,
                target: str = None,
                fx: dict = {},
            ) -> io.BytesIO:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        my_clip = mp.VideoFileClip(file_src)

        if "subclip" in fx.keys():
            my_clip = my_clip.subclip(*fx["subclip"])

        if fx.keys():
            my_clip.write_videofile(file_src)

        if my_clip.audio:
            audio_clip = mp.CompositeAudioClip([my_clip.audio])

        video_target_bgr = None

        if target_type == "color":
            video_target_bgr = self._generate_target_color(tempdir, my_clip.size, my_clip.duration, target)
        elif target_type == "image":
            video_target_bgr = self._generate_target_image(tempdir, my_clip.size, my_clip.duration, target)
        elif target_type == "video":
            video_target_bgr = self._generate_target_video(tempdir, my_clip.size, my_clip.duration, target)

        self.video_matte.matting(file_src, file_bgr, f"{tempdir}/output", video_target_bgr)

        video_clip = mp.VideoFileClip(f"{tempdir}/output/com.mp4")

        if my_clip.audio:
            video_clip.audio = audio_clip

        video_clip.write_videofile(f"{tempdir}/output/ok.mp4")

        with open(f"{tempdir}/output/ok.mp4","rb") as f:
            output_video = io.BytesIO(f.read())

        return output_video


