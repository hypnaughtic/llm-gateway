"""llm-gateway — Production-ready, vendor-agnostic LLM gateway.

Usage:
    from llm_gateway import LLMClient, LLMResponse, GatewayConfig

    llm = LLMClient()  # reads LLM_* env vars
    resp = await llm.complete(messages, response_model=MyModel)

Image generation:
    from llm_gateway import ImageClient, ImageGenerationResponse

    img = ImageClient()  # reads LLM_IMAGE_PROVIDER env var
    resp = await img.generate_image("a cat wearing a hat")
"""

from __future__ import annotations

from llm_gateway.client import LLMClient
from llm_gateway.config import GatewayConfig
from llm_gateway.cost import (
    CostTracker,
    ImageCostTracker,
    calculate_cost,
    calculate_image_cost,
    register_image_pricing,
    register_pricing,
)
from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderInitError,
    ProviderNotFoundError,
    ResponseValidationError,
)
from llm_gateway.image_client import ImageClient
from llm_gateway.providers.base import LLMProvider
from llm_gateway.providers.image_base import ImageGenerationProvider
from llm_gateway.registry import (
    build_image_provider,
    build_provider,
    list_image_providers,
    list_providers,
    register_image_provider,
    register_provider,
)
from llm_gateway.testing import (
    FakeCall,
    FakeImageCall,
    FakeImageProvider,
    FakeLLMProvider,
)
from llm_gateway.tokenizer import (
    Tokenizer,
    build_tokenizer,
    count_tokens,
    register_tokenizer,
)
from llm_gateway.types import (
    ImageData,
    ImageGenerationResponse,
    ImageTokenUsage,
    LLMMessage,
    LLMResponse,
    TokenUsage,
)

__all__ = [
    "CLINotFoundError",
    "CostLimitExceededError",
    "CostTracker",
    "FakeCall",
    "FakeImageCall",
    "FakeImageProvider",
    "FakeLLMProvider",
    "GatewayConfig",
    "GatewayError",
    "ImageClient",
    "ImageCostTracker",
    "ImageData",
    "ImageGenerationProvider",
    "ImageGenerationResponse",
    "ImageTokenUsage",
    "LLMClient",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "ProviderInitError",
    "ProviderNotFoundError",
    "ResponseValidationError",
    "TokenUsage",
    "Tokenizer",
    "build_image_provider",
    "build_provider",
    "build_tokenizer",
    "calculate_cost",
    "calculate_image_cost",
    "count_tokens",
    "list_image_providers",
    "list_providers",
    "register_image_pricing",
    "register_image_provider",
    "register_pricing",
    "register_provider",
    "register_tokenizer",
]
