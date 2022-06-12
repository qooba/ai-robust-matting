from fastapi import APIRouter, File, UploadFile, Request
from common import Injects
from starlette.responses import StreamingResponse
from inference_video import VideoMatteService
import io
from typing import List
import logging
router = APIRouter()

logger = logging.getLogger("mycoolapp")

@router.get("/api/matte")
async def root(matte_service: VideoMatteService = Injects(VideoMatteService)):
    video_src="/app/src.mp4"
    video_bgr="/app/bgr1.png"
    matte_service.process(video_src, video_bgr)
    return {"message": "ok"}

@router.post("/api/matte/{file_name}")
async def create_upload_file(file_name, request: Request, matte_service: VideoMatteService = Injects(VideoMatteService)):
    #print(file)
    #print(file_name)
    #print(file.filename)
    #file_data=await file.read()
    #video_src=f'/tmp/{file_name}'
    #with open(video_src, 'w') as f:
    #    f.write(file_data)

    #matte_service.process(video_src)
    #print(src.filename)

    form = await request.form()

    src = form["src"]
    print(src.filename)
    src_data = await src.read()

    bgr = form["bgr"]
    print(bgr.filename)
    bgr_data = await bgr.read()

    #return StreamingResponse(io.BytesIO(file_processed), media_type="video/mp4")
    return {"message": "OK"}
