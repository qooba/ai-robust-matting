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
    audio_clip = mp.CompositeAudioClip([my_clip.audio])
    #my_clip.audio.write_audiofile(f"{tempdir}/audio.mp3")

    bgr = form["bgr"]
    print(bgr.filename)
    bgr_data = await bgr.read()

    file_bgr = f'{tempdir}/{bgr.filename}'
    with open(file_bgr, "wb") as f:
        f.write(bgr_data)

    await ws_manager.send_text("Start Processing ⏳")
    matte_service.process(file_src, file_bgr, f"{tempdir}/output")


    await ws_manager.send_text("Appending audio 🎶")
    video_clip = mp.VideoFileClip(f"{tempdir}/output/com.mp4")
    video_clip.audio = audio_clip
    video_clip.write_videofile(f"{tempdir}/output/ok.mp4")

    with open(f"{tempdir}/output/ok.mp4","rb") as f:
        output_video = io.BytesIO(f.read())

    return StreamingResponse(output_video, media_type="video/mp4")
