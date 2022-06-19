from fastapi import APIRouter, File, UploadFile, Request, WebSocket
from common import Injects
from starlette.responses import StreamingResponse
from inference import VideoService
from services.ws import WebSocketManager
import io
import shutil
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
        del src_data
        del src

    target_type = form["targetType"]
    if target_type == "green":
        target = None
    else:
        target = form["target"]

    if target_type in ("image", "video"):
        target_data = await target.read()
        target_file = f'{tempdir}/{target.filename}'
        with open(target_file, "wb") as f:
            f.write(target_data)
            target = target_file
            del target_data

    fx_list = form["fx"]
    print(fx_list)

    fx = {}

    if "subclip" in fx_list:
        subclip_start = form["subclipStart"]
        subclip_end = form["subclipEnd"]
        fx["subclip"]=(subclip_start, subclip_end)
        print(subclip_start, subclip_end)

    await ws_manager.send_text("Start Processing ⏳")

    output_video = video_service.process(tempdir, file_src, target_type, target, fx)

    await ws_manager.send_text("Processing Finished ✅")

    #shutil.rmtree(tempdir)

    return StreamingResponse(output_video, media_type="video/mp4")
