"""
app/session/state.py — RealtimeSession 会话状态机。

实现协议核心逻辑：session.update / append / commit，
双缓冲（recv_buffer + ASRStreamingState）及 done 后立即 re-init。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Awaitable, Callable

from app.asr.adapter import QwenStreamingAdapter, SessionAsrConfig
from app.config import Settings
from app.protocol.errors import ErrorCode, build_error
from app.protocol.events import (
    AppendEvent,
    CommitEvent,
    SessionUpdateEvent,
    build_session_created,
    build_session_updated,
    build_transcription_delta,
    build_transcription_done,
)
from app.session.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)

# 向 WebSocket 发送 JSON 的回调类型
SendFn = Callable[[dict], Awaitable[None]]


class SessionPhase(str, Enum):
    """会话生命周期阶段。"""

    CONNECTED = "connected"
    CONFIGURED = "configured"


class RealtimeSession:
    """
    单个 WebSocket 连接的 Realtime ASR 会话。

    维护接收缓冲、Qwen3 流式 state 及会话配置；
    所有事件在 session 锁内串行处理。
    """

    def __init__(
        self,
        send: SendFn,
        asr: QwenStreamingAdapter,
        settings: Settings,
    ) -> None:
        """
        创建会话（尚未 init ASR state，需 session.update）。

        Args:
            send: 异步发送 JSON 到客户端的回调。
            asr: 全局 ASR 适配器。
            settings: 应用默认配置。
        """
        self.session_id = f"sess-{uuid.uuid4().hex[:12]}"
        self._send = send
        self._asr = asr
        self._settings = settings
        self._lock = asyncio.Lock()
        self._closed = False

        self.phase = SessionPhase.CONNECTED
        self.recv_buffer = AudioBuffer()
        self.asr_state: Any | None = None
        self.config = SessionAsrConfig(
            chunk_size_sec=settings.default_chunk_size_sec,
            unfixed_chunk_num=settings.default_unfixed_chunk_num,
            unfixed_token_num=settings.default_unfixed_token_num,
        )

    async def on_connect(self) -> None:
        """连接建立后发送 session.created。"""
        await self._send(build_session_created(self.session_id))

    async def handle(self, event: SessionUpdateEvent | AppendEvent | CommitEvent) -> None:
        """
        分发并处理客户端事件（同连接串行）。

        Args:
            event: 已校验的客户端事件。
        """
        if self._closed:
            return

        async with self._lock:
            if self._closed:
                return
            try:
                if isinstance(event, SessionUpdateEvent):
                    await self._handle_update(event)
                elif isinstance(event, AppendEvent):
                    await self._handle_append(event)
                elif isinstance(event, CommitEvent):
                    await self._handle_commit(event)
            except ValueError as exc:
                await self._send(build_error(str(exc), ErrorCode.INVALID_AUDIO))
            except Exception as exc:
                logger.exception("会话 %s 处理事件失败", self.session_id)
                await self._send(build_error(str(exc), ErrorCode.INFERENCE_ERROR))

    def close(self) -> None:
        """
        标记会话已关闭（断线 B 策略：不 finish，仅清理引用）。
        """
        self._closed = True
        self.recv_buffer.clear()
        self.asr_state = None

    async def _handle_update(self, event: SessionUpdateEvent) -> None:
        """
        处理 session.update：合并配置并 init_streaming_state。

        Args:
            event: session.update 事件。
        """
        if event.language is not None:
            self.config.language = event.language or None
        if event.hotwords is not None:
            self.config.hotwords = event.hotwords
        if event.chunk_size_sec is not None:
            self.config.chunk_size_sec = event.chunk_size_sec
        if event.unfixed_chunk_num is not None:
            self.config.unfixed_chunk_num = event.unfixed_chunk_num
        if event.unfixed_token_num is not None:
            self.config.unfixed_token_num = event.unfixed_token_num

        self.recv_buffer.clear()
        self.asr_state = self._asr.init_state(self.config)
        self.phase = SessionPhase.CONFIGURED

        await self._send(build_session_updated(self._config_dict()))
        logger.debug("会话 %s 已 update", self.session_id)

    async def _handle_append(self, event: AppendEvent) -> None:
        """
        处理 append：仅写入接收缓冲，不推理。

        Args:
            event: append 事件。
        """
        if self.asr_state is None:
            await self._send(
                build_error("请先发送 session.update", ErrorCode.SESSION_NOT_READY)
            )
            return

        self.recv_buffer.append_pcm16_b64(event.audio)
        self.phase = SessionPhase.CONFIGURED

    async def _handle_commit(self, event: CommitEvent) -> None:
        """
        处理 commit：drain 接收缓冲 → 推理 → 返回 delta 或 done。

        commit(true) 发完 done 后立即 re-init asr_state。

        Args:
            event: commit 事件。
        """
        if self.asr_state is None:
            await self._send(
                build_error("请先发送 session.update", ErrorCode.SESSION_NOT_READY)
            )
            return

        await self._commit(final=event.final)

    async def _commit(self, final: bool) -> None:
        """
        统一的 commit 实现。

        Args:
            final: True 表示本轮结束（finish + done + re-init）。
        """
        pcm = self.recv_buffer.drain()
        state = self.asr_state
        assert state is not None

        if pcm.size > 0:
            await self._asr.transcribe(pcm, state)

        if final:
            await self._asr.finish(state)
            await self._send(
                build_transcription_done(
                    text=getattr(state, "text", "") or "",
                    language=getattr(state, "language", "") or "",
                )
            )
            # done 后立即 init，同一 WS 可开始下一轮
            self.recv_buffer.clear()
            self.asr_state = self._asr.init_state(self.config)
            self.phase = SessionPhase.CONFIGURED
            logger.debug("会话 %s commit(true) 完成，已 re-init", self.session_id)
        else:
            await self._send(
                build_transcription_delta(
                    text=getattr(state, "text", "") or "",
                    language=getattr(state, "language", "") or "",
                )
            )

    def _config_dict(self) -> dict:
        """
        将会话配置序列化为 dict（用于 session.updated）。

        Returns:
            配置字典。
        """
        return {
            "language": self.config.language,
            "hotwords": self.config.hotwords,
            "chunk_size_sec": self.config.chunk_size_sec,
            "unfixed_chunk_num": self.config.unfixed_chunk_num,
            "unfixed_token_num": self.config.unfixed_token_num,
        }
