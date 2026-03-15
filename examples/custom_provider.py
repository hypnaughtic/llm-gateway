"""Demonstrates registering a custom provider."""

import asyncio
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    LLMClient,
    LLMResponse,
    TokenUsage,
    register_provider,
)
from llm_gateway.types import LLMMessage

T = TypeVar("T")


class EchoProvider:
    """A demo provider that echoes back the last user message."""

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "EchoProvider":
        return cls()

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        image_files: Sequence[str] | None = None,
    ) -> LLMResponse[T]:
        effective_model = model or "echo-v1"
        last_msg = messages[-1]["content"] if messages else "empty"
        content = response_model.model_validate({"text": f"Echo: {last_msg}"})  # type: ignore[attr-defined]
        return LLMResponse(
            content=content,
            usage=TokenUsage(input_tokens=len(str(messages)), output_tokens=len(str(content))),
            model=effective_model,
            provider="echo",
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    async def close(self) -> None:
        pass


class EchoAnswer(BaseModel):
    text: str


async def main() -> None:
    # Register the custom provider
    register_provider("echo", EchoProvider.from_config)

    # Use it via config
    config = GatewayConfig(provider="echo")
    async with LLMClient(config=config) as llm:
        resp = await llm.complete(
            messages=[{"role": "user", "content": "Hello, world!"}],
            response_model=EchoAnswer,
        )
        print(f"Provider: {resp.provider}")
        print(f"Response: {resp.content.text}")


if __name__ == "__main__":
    asyncio.run(main())
