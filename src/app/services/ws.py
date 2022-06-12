import logging
from fastapi import WebSocket

class WebSocketManager:
    sockets = None

    async def send_text(self, message: str):
        socket = WebSocketManager.sockets
        try:
            await socket.send_text(message)
        except Exception as ex:
            logging.info(f'socket deleted because of exception: {ex}')
            print(ex)
            WebSocketManager.sockets = None


    async def add(self, socket: WebSocket):
        WebSocketManager.sockets = socket
