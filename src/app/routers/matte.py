from fastapi import APIRouter, File, UploadFile, Request, WebSocket
from common import Injects
from starlette.responses import StreamingResponse
from inference_video import VideoService
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
                             video_service: VideoService = Injects(VideoService),
                             ws_manager: WebSocketManager = Injects(WebSocketManager)
                             ):

    tempdir = tempfile.mkdtemp()

    form = await request.form()

    src = form["src"]
    src_data = await src.read()

    file_src = f'{tempdir}/{src.filename}'
    with open(file_src, "wb") as f:
        f.write(src_data)

    bgr = form["bgr"]
    bgr_data = await bgr.read()

    file_bgr = f'{tempdir}/{bgr.filename}'
    with open(file_bgr, "wb") as f:
        f.write(bgr_data)

    await ws_manager.send_text("Start Processing ⏳")

    output_video = video_service.process(tempdir, file_src, file_bgr)

    await ws_manager.send_text("Processing Finished ✅")

    return StreamingResponse(output_video, media_type="video/mp4")
