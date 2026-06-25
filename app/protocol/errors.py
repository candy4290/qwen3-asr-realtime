"""
app/protocol/errors.py — 协议错误码与错误事件构建。

统一 error 事件的 code 枚举及 build_error 工厂函数。
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    """WebSocket 协议错误码枚举。"""

    INVALID_MESSAGE = "invalid_message"
    INVALID_AUDIO = "invalid_audio"
    SESSION_NOT_READY = "session_not_ready"
    INFERENCE_ERROR = "inference_error"
    INTERNAL_ERROR = "internal_error"


def build_error(message: str, code: ErrorCode = ErrorCode.INTERNAL_ERROR) -> dict:
    """
    构建服务端 error 事件 JSON 对象。

    Args:
        message: 人类可读的错误描述。
        code: 机器可读的错误码。

    Returns:
        可直接 json.dumps 发送的字典。
    """
    return {
        "type": "error",
        "error": message,
        "code": code.value,
    }
