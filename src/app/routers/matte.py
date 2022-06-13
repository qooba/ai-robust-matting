from fastapi import APIRouter, File, UploadFile, Request, WebSocket
from common import Injects
from starlette.responses import StreamingResponse
from inference_video import VideoMatteService
from services.ws import WebSocketManager
import io
from typing import List
import logging
import tempfile
import moviepy.editor as mp
from PIL import Image

router = APIRouter()

logger = logging.getLogger("mycoolapp")


@router.post("/api/matte")
async def create_upload_file(request: Request,
                             matte_service: VideoMatteService = Injects(VideoMatteService),
                             ws_manager: WebSocketManager = Injects(WebSocketManager)
                             ):

    tempdir = tempfile.mkdtemp()

    form = await request.form()

    src = form["src"]
    print(src.filename)
    src_data = await src.read()

    file_src = f'{tempdir}/{src.filename}'
    with open(file_src, "wb") as f:
        f.write(src_data)

    await ws_manager.send_text("Extracting audio 🎶")
    my_clip = mp.VideoFileClip(file_src)

    if my_clip.audio:
        audio_clip = mp.CompositeAudioClip([my_clip.audio])

    bgr = form["bgr"]
    print(bgr.filename)
    bgr_data = await bgr.read()

    file_bgr = f'{tempdir}/{bgr.filename}'
    with open(file_bgr, "wb") as f:
        f.write(bgr_data)

    await ws_manager.send_text("Start Processing ⏳")

    #use image background
    video_target_bgr=f"{tempdir}/video_target_bgr.mp4"
    video_target_bgr_img=f"{tempdir}/video_target_bgr.png"
    ##clips = [mp.ImageClip("/app/tmp/bgr_white.png").set_duration(my_clip.duration)]
    Image.new('RGB', my_clip.size, color = (73, 109, 137)).save(video_target_bgr_img)
    clips = [mp.ImageClip(video_target_bgr_img).set_duration(my_clip.duration)]

    concat_clip = mp.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_target_bgr, fps=24)

    matte_service.process(file_src, file_bgr, f"{tempdir}/output", video_target_bgr)

    await ws_manager.send_text("Appending audio 🎶")
    video_clip = mp.VideoFileClip(f"{tempdir}/output/com.mp4")

    if my_clip.audio:
        video_clip.audio = audio_clip

    video_clip.write_videofile(f"{tempdir}/output/ok.mp4")

    with open(f"{tempdir}/output/ok.mp4","rb") as f:
        output_video = io.BytesIO(f.read())

    return StreamingResponse(output_video, media_type="video/mp4")
