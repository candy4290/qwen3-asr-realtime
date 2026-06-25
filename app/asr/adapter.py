"""
app/asr/adapter.py — Qwen3 流式 ASR 适配器。

负责：
- 启动时加载 Qwen3ASRModel.LLM（vLLM 后端）
- init_state / transcribe / finish 封装
- 通过 asyncio.to_thread 避免阻塞事件循环
- 模型级 asyncio.Lock 保证 vLLM 串行推理
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SessionAsrConfig:
    """传给 init_streaming_state 的会话级 ASR 参数。"""

    language: str | None = None
    hotwords: str = ""
    chunk_size_sec: float = 1.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5


class QwenStreamingAdapter:
    """
    Qwen3-ASR 流式推理适配器。

    全局单例，多 WebSocket 连接共享同一 vLLM 模型实例。
    """

    def __init__(self) -> None:
        """初始化适配器；模型在 load() 中延迟加载。"""
        self._model: Any = None
        self._infer_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        """模型是否已成功加载。"""
        return self._model is not None

    def _check_vllm_version(self) -> None:
        """
        检查 vLLM 版本是否与 Qwen3-ASR 兼容。

        vLLM 0.16+ 移除了 BaseMultiModalProcessor._get_data_parser，
        会导致 Qwen3-ASR 在 renderers/hf.py 处启动失败。
        """
        try:
            import vllm
        except ImportError as exc:
            raise RuntimeError(
                "未安装 vLLM，请先执行: pip install -U qwen-asr[vllm]"
            ) from exc

        version = getattr(vllm, "__version__", "0.0.0")
        parts = version.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        if (major, minor) >= (0, 16):
            raise RuntimeError(
                f"当前 vLLM {version} 与 Qwen3-ASR 不兼容（需 < 0.16）。"
                "请执行: pip install 'vllm>=0.14.0,<0.16.0'"
            )

    def load(self, settings: Settings) -> None:
        """
        加载 Qwen3ASRModel（vLLM 后端）。

        Args:
            settings: 应用配置，含模型路径与 GPU 参数。

        Raises:
            RuntimeError: 重复加载时抛出。
        """
        if self._model is not None:
            raise RuntimeError("ASR 模型已加载，不可重复 load")

        self._check_vllm_version()

        from qwen_asr import Qwen3ASRModel

        logger.info("正在加载 ASR 模型: %s", settings.asr_model_path)
        self._model = Qwen3ASRModel.LLM(
            model=settings.asr_model_path,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_len=settings.max_model_len,
            max_new_tokens=settings.max_new_tokens,
        )
        logger.info("ASR 模型加载完成")

    def init_state(self, config: SessionAsrConfig) -> Any:
        """
        创建新的流式 ASR 状态（同步，耗时极短）。

        Args:
            config: 会话 ASR 配置。

        Returns:
            Qwen3 ASRStreamingState 对象。
        """
        self._ensure_loaded()
        return self._model.init_streaming_state(
            context=config.hotwords or "",
            language=config.language,
            unfixed_chunk_num=config.unfixed_chunk_num,
            unfixed_token_num=config.unfixed_token_num,
            chunk_size_sec=config.chunk_size_sec,
        )

    async def transcribe(self, pcm: np.ndarray, state: Any) -> None:
        """
        流式识别一步：将 PCM 送入 streaming_transcribe。

        在线程池中执行，并持有模型锁。

        Args:
            pcm: 16kHz 单声道 PCM（int16 或 float32）。
            state: ASRStreamingState。
        """
        if pcm.size == 0:
            return
        self._ensure_loaded()
        async with self._infer_lock:
            await asyncio.to_thread(self._model.streaming_transcribe, pcm, state)

    async def finish(self, state: Any) -> None:
        """
        流式识别收尾：flush 不足一个 chunk 的尾部音频。

        在线程池中执行，并持有模型锁。

        Args:
            state: ASRStreamingState。
        """
        self._ensure_loaded()
        async with self._infer_lock:
            await asyncio.to_thread(self._model.finish_streaming_transcribe, state)

    def _ensure_loaded(self) -> None:
        """确认模型已加载，否则抛出 RuntimeError。"""
        if self._model is None:
            raise RuntimeError("ASR 模型尚未加载")
