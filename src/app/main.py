from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from common import Bootstapper, Injects
from routers import matte
from services.ws import WebSocketManager


app = FastAPI()
container=Bootstapper().bootstrap()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, ws_manager: WebSocketManager = Injects(WebSocketManager)):
    await websocket.accept()
    await ws_manager.add(websocket)

    while True:
        await websocket.receive_text()

@app.get("/", response_class=HTMLResponse)
async def homepage(include_in_schema=False):
    return FileResponse("static/index.html")

app.include_router(matte.router, tags=['matte'])
