"""
食刻推荐 - LangChain + LangGraph workflow for Xiaohongshu posting.

Note: Lazily import heavy modules to avoid circular imports when running submodules directly.
"""


def build_app():
    from .workflow import build_app as _build_app
    return _build_app()


def run_workflow():
    from .workflow import run_workflow as _run_workflow
    return _run_workflow()


__all__ = ["build_app", "run_workflow"]
