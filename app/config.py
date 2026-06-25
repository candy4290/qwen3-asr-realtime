"""
app/config.py — 应用配置模块。

通过环境变量或 .env 文件加载服务运行参数，包括模型路径、GPU 显存占用、
默认流式 chunk 参数及 HTTP/WS 监听地址。
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务全局配置，支持环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ASR 模型路径（HuggingFace repo id 或本地目录）
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    # vLLM GPU 显存占用比例（相对单卡总显存）
    gpu_memory_utilization: float = 0.8
    # vLLM 最大上下文长度；模型默认 65536 会占用大量 KV cache，实时 ASR 无需这么长
    max_model_len: int = 8192
    # 流式推理单次最大生成 token 数（流式场景建议较小）
    max_new_tokens: int = 32

    # HTTP / WebSocket 监听地址
    host: str = "0.0.0.0"
    port: int = 9800

    # 为 True 时设置 HF_HUB_OFFLINE=1，仅从本地 HuggingFace 缓存加载模型
    hf_hub_offline: bool = False

    # session.update 未指定时使用的默认流式参数
    default_chunk_size_sec: float = 1.0
    default_unfixed_chunk_num: int = 2
    default_unfixed_token_num: int = 5


# 全局单例配置对象
settings = Settings()

# HuggingFace Hub 离线开关（需在 import qwen_asr 之前生效）
if settings.hf_hub_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
