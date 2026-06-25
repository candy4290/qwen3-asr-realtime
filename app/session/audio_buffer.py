"""
app/session/audio_buffer.py — 音频接收缓冲。

负责 PCM16 base64 解码、样本追加与一次性 drain（供 commit 送入推理）。
"""

from __future__ import annotations

import base64
import binascii

import numpy as np


class AudioBuffer:
    """
    客户端 append 的 PCM16 音频接收缓冲。

    仅负责存储与取出，不进行推理。
    """

    def __init__(self) -> None:
        """创建空的 float32 样本缓冲（与 Qwen3 streaming_transcribe 兼容）。"""
        self._samples = np.zeros((0,), dtype=np.float32)

    @property
    def size(self) -> int:
        """当前缓冲中的采样点数。"""
        return int(self._samples.shape[0])

    def append_pcm16_b64(self, audio_b64: str) -> None:
        """
        解码 base64 PCM16 并追加到缓冲。

        Args:
            audio_b64: PCM16 little-endian 的 base64 字符串。

        Raises:
            ValueError: base64 非法或字节长度不是 2 的倍数。
        """
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"无效的 base64 音频: {exc}") from exc

        if len(raw) % 2 != 0:
            raise ValueError("PCM16 字节长度必须为 2 的倍数")

        if len(raw) == 0:
            return

        int16 = np.frombuffer(raw, dtype=np.int16)
        float32 = (int16.astype(np.float32) / 32768.0)
        self._samples = np.concatenate([self._samples, float32])

    def drain(self) -> np.ndarray:
        """
        取出并清空全部缓冲样本。

        Returns:
            float32 一维数组；若为空则返回长度为 0 的数组。
        """
        out = self._samples
        self._samples = np.zeros((0,), dtype=np.float32)
        return out

    def clear(self) -> None:
        """清空缓冲，不返回数据。"""
        self._samples = np.zeros((0,), dtype=np.float32)
