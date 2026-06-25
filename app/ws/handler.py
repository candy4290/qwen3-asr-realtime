"""
app/ws/handler.py — WebSocket 路由处理器。

职责：
- 接受 /v1/realtime 连接
- 解析 JSON 并用 Pydantic 校验客户端事件
- 委托 RealtimeSession 处理业务逻辑
- 断线时静默清理会话（不 finish）
"""

from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.asr.adapter import QwenStreamingAdapter
from app.config import Settings
from app.protocol.errors import ErrorCode, build_error
from app.protocol.events import ClientEvent
from app.session.manager import SessionManager
from app.session.state import RealtimeSession

logger = logging.getLogger(__name__)

# Pydantic  discriminated union 解析器
_client_event_adapter = TypeAdapter(ClientEvent)


async def realtime_websocket(
    websocket: WebSocket,
    asr: QwenStreamingAdapter,
    settings: Settings,
    session_manager: SessionManager,
) -> None:
    """
    WebSocket 主循环：/v1/realtime。

    Args:
        websocket: FastAPI WebSocket 实例。
        asr: 全局 ASR 适配器。
        settings: 应用配置。
        session_manager: 会话管理器。
    """
    await websocket.accept()

    async def send_json(data: dict) -> None:
        """向当前连接发送 JSON 消息。"""
        await websocket.send_json(data)

    session = RealtimeSession(send=send_json, asr=asr, settings=settings)
    session_manager.register(session)

    try:
        await session.on_connect()

        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(
                    build_error("无效的 JSON", ErrorCode.INVALID_MESSAGE)
                )
                continue

            try:
                event = _client_event_adapter.validate_python(payload)
            except ValidationError as exc:
                await send_json(
                    build_error(
                        f"事件格式错误: {exc.errors()[0]['msg']}",
                        ErrorCode.INVALID_MESSAGE,
                    )
                )
                continue

            await session.handle(event)

    except WebSocketDisconnect:
        logger.info("WebSocket 断开: %s", session.session_id)
    except Exception:
        logger.exception("WebSocket 异常: %s", session.session_id)
    finally:
        session.close()
        session_manager.remove(session)
