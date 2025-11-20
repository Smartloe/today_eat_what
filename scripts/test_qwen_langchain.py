"""
Quick LangChain call to SiliconFlow Qwen/Qwen3-8B.

Usage:
    SILICONFLOW_API_KEY=sk-... uv run python scripts/test_qwen_langchain.py
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("请在环境变量中设置 SILICONFLOW_API_KEY 或 QWEN_API_KEY")

    llm = ChatOpenAI(
        model="Qwen/Qwen3-8B",
        api_key=api_key,
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是行业分析师，请用简洁中文回答。"),
            (
                "human",
                "中国大模型行业在 2025 年会面临哪些机会和挑战？用 120 字以内回答，并给出 2 个要点列表。",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({})
    print(result.content)


if __name__ == "__main__":
    main()
