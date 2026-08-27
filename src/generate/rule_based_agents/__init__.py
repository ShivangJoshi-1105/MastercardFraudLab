"""Importing this package registers all 10 attack agents via their @register_agent decorators."""

from . import tabular_agents  # noqa: F401
from . import graph_agents  # noqa: F401
from .base import AgentContext, AttackAgent, all_agents, get_agent, graph_agents as graph_agent_registry, tabular_agents as tabular_agent_registry

__all__ = [
    "AgentContext",
    "AttackAgent",
    "all_agents",
    "get_agent",
    "graph_agent_registry",
    "tabular_agent_registry",
]
