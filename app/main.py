"""
app/main.py — FastAPI 应用入口。

职责：
- lifespan 中加载 Qwen3 ASR 模型
- 挂载 WebSocket /v1/realtime 与 HTTP /health
- 提供 static/demo.html 测试页
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.asr.adapter import QwenStreamingAdapter
from app.config import settings
from app.session.manager import SessionManager
from app.ws.handler import realtime_websocket

# 日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 项目根目录（用于 static）
ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期：启动时加载 ASR 模型，关闭时记录日志。

    Args:
        app: FastAPI 应用实例。
    """
    asr = QwenStreamingAdapter()
    asr.load(settings)
    app.state.asr = asr
    app.state.session_manager = SessionManager()
    logger.info("服务已启动，监听 %s:%s", settings.host, settings.port)
    yield
    logger.info("服务正在关闭")


app = FastAPI(
    title="Qwen3-ASR Realtime",
    description="基于 WebSocket 的 Qwen3-ASR 流式语音识别服务",
    lifespan=lifespan,
)

# 静态资源（除 demo 根路径外）
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    """
    健康检查接口。

    Returns:
        服务状态、模型是否已加载、当前活跃 WS 会话数。
    """
    asr: QwenStreamingAdapter = app.state.asr
    manager: SessionManager = app.state.session_manager
    return {
        "status": "ok",
        "model_loaded": asr.is_loaded,
        "active_sessions": manager.active_count,
    }


@app.get("/demo.html")
async def demo_page():
    """
    返回内置 HTML 测试页。

    Returns:
        demo.html 文件响应。
    """
    path = STATIC_DIR / "demo.html"
    return FileResponse(path)


@app.websocket("/v1/realtime")
async def ws_realtime(websocket: WebSocket):
    """
    Realtime ASR WebSocket 端点。

    Args:
        websocket: 客户端 WebSocket 连接。
    """
    await realtime_websocket(
        websocket=websocket,
        asr=app.state.asr,
        settings=settings,
        session_manager=app.state.session_manager,
    )


def main():
    """命令行启动入口（python -m app.main）。"""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
