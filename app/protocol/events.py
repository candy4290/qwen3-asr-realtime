"""
app/protocol/events.py — Realtime ASR WebSocket 事件模型。

定义客户端入站事件（session.update / append / commit）的 Pydantic 校验模型，
以及服务端出站事件（session.created / delta / done 等）的构建函数。
"""

import time
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 客户端入站事件
# ---------------------------------------------------------------------------


class SessionUpdateEvent(BaseModel):
    """session.update：配置识别参数并触发 init_streaming_state。"""

    type: Literal["session.update"] = "session.update"
    language: str | None = None
    hotwords: str = ""
    chunk_size_sec: float | None = None
    unfixed_chunk_num: int | None = None
    unfixed_token_num: int | None = None


class AppendEvent(BaseModel):
    """input_audio_buffer.append：追加 PCM16 base64 音频到接收缓冲。"""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str


class CommitEvent(BaseModel):
    """input_audio_buffer.commit：提交接收缓冲进行推理。"""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"
    final: bool = False


ClientEvent = Annotated[
    Union[SessionUpdateEvent, AppendEvent, CommitEvent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# 服务端出站事件构建
# ---------------------------------------------------------------------------

# 固定音频格式约定（写入 session.created）
AUDIO_FORMAT = {
    "format": "pcm16",
    "sample_rate": 16000,
    "channels": 1,
}


def build_session_created(session_id: str) -> dict:
    """
    构建 session.created 事件。

    Args:
        session_id: 本会话唯一标识。

    Returns:
        session.created 事件字典。
    """
    return {
        "type": "session.created",
        "id": session_id,
        "created": int(time.time()),
        "audio": AUDIO_FORMAT,
    }


def build_session_updated(config: dict) -> dict:
    """
    构建 session.updated 事件，回显当前会话配置。

    Args:
        config: 当前 SessionConfig 序列化字典。

    Returns:
        session.updated 事件字典。
    """
    return {
        "type": "session.updated",
        "session": config,
    }


def build_transcription_delta(text: str, language: str) -> dict:
    """
    构建 transcription.delta 事件（累计全文）。

    Args:
        text: 当前累计识别文本。
        language: 当前识别语种。

    Returns:
        transcription.delta 事件字典。
    """
    return {
        "type": "transcription.delta",
        "text": text,
        "language": language,
    }


def build_transcription_done(text: str, language: str) -> dict:
    """
    构建 transcription.done 事件（本轮最终结果）。

    Args:
        text: 累计识别文本。
        language: 识别语种。

    Returns:
        transcription.done 事件字典。
    """
    return {
        "type": "transcription.done",
        "text": text,
        "language": language,
    }
