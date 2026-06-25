"""
app/asr/engine_watchdog.py — vLLM EngineCore 存活检测。

Qwen3ASRModel.LLM 底层使用 vLLM V1 架构，推理在独立 EngineCore 子进程中运行。
当 EngineCore OOM、崩溃或被 kill 时，vLLM 会将 engine_dead 置为 True，
但不会自动退出 FastAPI 主进程；本模块负责探测该状态。
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def fatal_shutdown(reason: str, exit_code: int = 1) -> None:
    """
    记录致命错误并立即终止整个进程。

    EngineCore 挂掉后 FastAPI 无法继续提供 ASR 服务，交由外部进程管理器重启。
    """
    logger.critical("%s，主进程即将关闭（请检查 GPU/OOM 日志后重启）", reason)
    os._exit(exit_code)


def get_engine_core_client(qwen_asr_model: Any) -> Any | None:
    """
    从 **本程序加载的** Qwen3ASRModel 实例取出 vLLM EngineCoreClient。

    注意：此处不做系统级进程扫描（不会 ps/grep 所有 EngineCore），
    只沿内存对象链取到 launch_core_engines() 为本实例创建的 client。

    访问链：Qwen3ASRModel.model → vllm.LLM.llm_engine → engine_core

    Args:
        qwen_asr_model: 本服务 lifespan 中 load() 得到的 Qwen3ASRModel 实例。

    Returns:
        EngineCoreClient，无法解析时返回 None。
    """
    if getattr(qwen_asr_model, "backend", None) != "vllm":
        return None

    vllm_llm = getattr(qwen_asr_model, "model", None)
    if vllm_llm is None:
        return None

    llm_engine = getattr(vllm_llm, "llm_engine", None)
    if llm_engine is None:
        return None

    return getattr(llm_engine, "engine_core", None)


def get_monitored_engine_pids(qwen_asr_model: Any) -> list[int]:
    """
    返回本实例 vLLM 在 load 时拉起的 EngineCore 子进程 PID 列表。

    来源是 engine_manager.processes 里保存的 Process 句柄，与 vLLM 内部
    MPClientEngineMonitor 使用的是同一组对象，不会误匹配系统中其他 vLLM 服务。

    Args:
        qwen_asr_model: Qwen3ASRModel 实例。

    Returns:
        子进程 PID 列表；无法解析时返回空列表。
    """
    core = get_engine_core_client(qwen_asr_model)
    if core is None:
        return []

    resources = getattr(core, "resources", None)
    engine_manager = getattr(resources, "engine_manager", None) if resources else None
    processes = getattr(engine_manager, "processes", None) if engine_manager else None
    if not processes:
        return []

    pids: list[int] = []
    for proc in processes:
        pid = getattr(proc, "pid", None)
        if pid is not None:
            pids.append(int(pid))
    return pids


def is_vllm_engine_alive(qwen_asr_model: Any) -> bool:
    """
    检测 **本程序拉起的** vLLM EngineCore 是否仍在运行。

    检测对象均来自本 Qwen3ASRModel 实例持有的 client/resources，包括：
    - engine_dead：vLLM MPClientEngineMonitor 对本 client 维护的标志
    - engine_manager.processes：load 时 launch_core_engines 返回的 Process 句柄
    - ensure_alive()：向本 client 绑定的 ZMQ EngineCore 发存活检查

    不会扫描或匹配系统中其他 vLLM / EngineCore 进程。

    Args:
        qwen_asr_model: Qwen3ASRModel 实例。

    Returns:
        True 表示引擎正常；False 表示本实例的 EngineCore 已挂。
    """
    core = get_engine_core_client(qwen_asr_model)
    if core is None:
        logger.warning("无法获取 vLLM EngineCoreClient，跳过存活检测")
        return True

    resources = getattr(core, "resources", None)
    if resources is not None and getattr(resources, "engine_dead", False):
        return False

    engine_manager = getattr(resources, "engine_manager", None) if resources else None
    processes = getattr(engine_manager, "processes", None) if engine_manager else None
    if processes:
        dead = [
            f"pid={proc.pid} name={proc.name}"
            for proc in processes
            if not proc.is_alive()
        ]
        if dead:
            logger.error("本实例 EngineCore 子进程已退出: %s", ", ".join(dead))
            return False

    # 调用 vLLM 内置 ensure_alive，会抛出 EngineDeadError
    ensure_alive = getattr(core, "ensure_alive", None)
    if ensure_alive is not None:
        try:
            ensure_alive()
        except Exception:
            return False

    return True
