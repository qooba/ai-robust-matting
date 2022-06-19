"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
import io
import moviepy.editor as mp
from moviepy.video.fx.all import crop, resize
from PIL import Image, ImageColor
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from model import MattingNetwork

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):

    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """

    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:

                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])

                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)

class VideoService:

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

    def _find_sizes(self, source_size, target_size):

        s_width, s_height = source_size
        t_width, t_height = target_size

        new_width = t_width
        new_height = int(new_width * s_height/s_width)
        left=0
        right=t_width
        top = (new_height - t_height)/2
        bottom = top+t_height

        if new_height < t_height:
            new_height = t_height
            new_width = int(new_height * s_width/s_height)
            top=0
            bottom=t_height
            left = (new_width - t_width)/2
            right = left+t_width

        return (new_width, new_height), (left, top, right, bottom)

    def _generate_target_image(self, tempdir, size, duration, target_image):
        video_target_bgr=f"{tempdir}/video_target_bgr.mp4"
        video_target_bgr_img=f"{tempdir}/video_target_bgr.png"

        im = Image.open(target_image)

        new_size, crop_size = self._find_sizes(im.size, size)

        im = im.resize(new_size, Image.ANTIALIAS)
        im.crop(crop_size).save(video_target_bgr_img)
        clips = [mp.ImageClip(video_target_bgr_img).set_duration(duration)]

        concat_clip = mp.concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_target_bgr, fps=24)
        return video_target_bgr

    def _generate_target_video(self, tempdir, size, duration, target_video):
        clip = mp.VideoFileClip(target_video)
        video_target_bgr=f"{tempdir}/video_target_bgr.mp4"

        new_size, crop_size = self._find_sizes(clip.size, size)
        crop(resize(clip, new_size), x1=crop_size[0], y1=crop_size[1], x2=crop_size[2], y2=crop_size[3]).write_videofile(video_target_bgr)
        return video_target_bgr

    def process(self,
                tempdir: str,
                file_src: str,
                target_type: str,
                target: str = None,
                fx: dict = {},
                variant: str = 'mobilenetv3', #'mobilenetv3', 'resnet50'
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

        output_alpha=f"{tempdir}/pha.mp4"
        output_composition = None
        output_src=file_src

        if target_type == "color":
            video_target_bgr = self._generate_target_color(tempdir, my_clip.size, my_clip.duration, target)
        elif target_type == "image":
            video_target_bgr = self._generate_target_image(tempdir, my_clip.size, my_clip.duration, target)
        elif target_type == "video":
            video_target_bgr = self._generate_target_video(tempdir, my_clip.size, my_clip.duration, target)
        else:
            output_composition=f"{tempdir}/com.mp4"
            output_src=output_composition


        if variant == 'mobilenetv3':
            model = MattingNetwork('mobilenetv3').eval()
            model = model.cuda() if device == 'cuda' else model
            model.load_state_dict(torch.load('/app/models/rvm_mobilenetv3.pth'))
        elif variant == 'resnet50':
            model = MattingNetwork('resnet50').eval()
            model = model.cuda() if device == 'cuda' else model
            model.load_state_dict(torch.load('/app/models/rvm_resnet50.pth'))
        else:
            raise ValueError('Wrong variant please use: mobilenetv3 or resnet50')


        convert_video(
            model,                                             # The model, can be on any device (cpu or cuda).
            input_source=file_src,                             # A video file or an image sequence directory.
            output_type='video',                               # Choose "video" or "png_sequence"
            output_composition=output_composition,             # File path if video; directory path if png sequence.
            output_alpha=output_alpha,                         # [Optional] Output the raw alpha prediction.
            output_foreground=None,                            # [Optional] Output the raw foreground prediction.
            output_video_mbps=4,                               # Output video mbps. Not needed for png sequence.
            downsample_ratio=None,                             # A hyperparameter to adjust or use None for auto.
            seq_chunk=5,                                       # Process n frames at once for better parallelism.
        )

        if target_type != 'green':
            mask_clip = VideoFileClip(output_alpha, ismask=True)
            background_clip = VideoFileClip(video_target_bgr)
            video_clip = mp.VideoFileClip(file_src)
            video_clip.set_mask(mask_clip)
            ok_clip = mp.mask_clip([
                background_clip,
                video_clip
            ])

            del video_clip
            del background_clip
        else:
            ok_clip = mp.VideoFileClip(output_composition)


        if my_clip.audio:
            ok_clip.audio = audio_clip

        ok_clip.write_videofile(f"{tempdir}/ok.mp4")

        del ok_clip
        del my_clip
        del audio_clip

        with open(f"{tempdir}/ok.mp4","rb") as f:
            output_video = io.BytesIO(f.read())

        return output_video




