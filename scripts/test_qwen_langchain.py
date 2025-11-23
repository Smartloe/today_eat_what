"""
Quick LangChain call to SiliconFlow Qwen (model name from env).

Usage:
    Qwen_API_KEY=sk-... QWEN_MODEL=<model-id> uv run python scripts/test_qwen_langchain.py
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from today_eat_what.config import QWEN_MODEL_DEFAULT, Qwen_BASE_URL


def main() -> None:
    api_key = os.environ.get("Qwen_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("请在环境变量中设置 Qwen_API_KEY 或 QWEN_API_KEY")

    model_name = QWEN_MODEL_DEFAULT
    if not model_name:
        raise SystemExit("请在环境变量中设置 QWEN_MODEL 或 Qwen_MODEL")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=Qwen_BASE_URL,
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
