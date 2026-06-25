"""
app/session/manager.py — 会话管理器。

跟踪所有活跃 WebSocket 会话，供健康检查统计与断线清理。
"""

from __future__ import annotations

from app.session.state import RealtimeSession


class SessionManager:
    """
    全局会话注册表。

    每个 WebSocket 连接对应一个 RealtimeSession。
    """

    def __init__(self) -> None:
        """初始化空会话集合。"""
        self._sessions: set[RealtimeSession] = set()

    @property
    def active_count(self) -> int:
        """当前活跃会话数量。"""
        return len(self._sessions)

    def register(self, session: RealtimeSession) -> None:
        """
        注册新会话。

        Args:
            session: 刚创建的 RealtimeSession。
        """
        self._sessions.add(session)

    def remove(self, session: RealtimeSession) -> None:
        """
        移除会话（断线时调用，不触发 finish 推理）。

        Args:
            session: 待移除的会话。
        """
        self._sessions.discard(session)
