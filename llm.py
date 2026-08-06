import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def build_llm(temperature: float = 0) -> "ChatOpenAI":
    """Build a chat model pointed at the gateway, configured from .env."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError(
            "langchain-openai is not installed; install the openai extra: "
            "uv sync --extra openai"
        ) from error

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    if not model:
        raise RuntimeError("OPENAI_MODEL is missing from .env")
    if not base_url:
        raise RuntimeError("BASE_URL is missing from .env")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        use_responses_api=True,
    )