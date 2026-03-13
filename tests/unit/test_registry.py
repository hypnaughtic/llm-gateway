"""Tests for the provider registry."""

from __future__ import annotations

import builtins
import sys
from unittest.mock import MagicMock

import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import ProviderInitError, ProviderNotFoundError
from llm_gateway.registry import (
    _IMAGE_PROVIDERS,
    _PROVIDERS,
    build_image_provider,
    build_provider,
    list_image_providers,
    list_providers,
    register_image_provider,
    register_provider,
)


@pytest.mark.unit
class TestRegistry:
    def test_register_and_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_provider = MagicMock()
        factory = MagicMock(return_value=mock_provider)

        register_provider("test_provider", factory)

        monkeypatch.setenv("LLM_PROVIDER", "test_provider")
        config = GatewayConfig()
        result = build_provider(config)

        factory.assert_called_once_with(config)
        assert result is mock_provider

        # Cleanup
        _PROVIDERS.pop("test_provider", None)

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "nonexistent_xyz_999")
        config = GatewayConfig()
        with pytest.raises(ProviderNotFoundError, match="nonexistent_xyz_999"):
            build_provider(config)

    def test_list_providers(self) -> None:
        providers = list_providers()
        assert isinstance(providers, list)
        # At minimum, anthropic should be available if installed

    def test_factory_error_raises_provider_init_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ProviderInitError raised when factory function throws."""

        def bad_factory(config: GatewayConfig) -> None:
            raise RuntimeError("factory exploded")

        register_provider("broken_provider", bad_factory)  # type: ignore[arg-type]

        monkeypatch.setenv("LLM_PROVIDER", "broken_provider")
        config = GatewayConfig()
        with pytest.raises(ProviderInitError, match="broken_provider"):
            build_provider(config)

        # Cleanup
        _PROVIDERS.pop("broken_provider", None)


@pytest.mark.unit
class TestImageProviderRegistry:
    """Tests for the image provider registry functions."""

    def test_register_and_build_image_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Register an image provider factory and build it from config."""
        import llm_gateway.registry as reg

        # Reset image builtins flag so _ensure_image_builtins_registered runs fresh
        orig_flag = reg._image_builtins_registered
        reg._image_builtins_registered = True  # skip builtins, test our own

        mock_provider = MagicMock()
        factory = MagicMock(return_value=mock_provider)

        register_image_provider("test_image_provider", factory)

        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "test_image_provider")
        config = GatewayConfig()
        result = build_image_provider(config)

        factory.assert_called_once_with(config)
        assert result is mock_provider

        # Cleanup
        _IMAGE_PROVIDERS.pop("test_image_provider", None)
        reg._image_builtins_registered = orig_flag

    def test_unknown_image_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ProviderNotFoundError raised for unregistered image provider."""
        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "nonexistent_img_999")
        config = GatewayConfig()
        with pytest.raises(ProviderNotFoundError, match="nonexistent_img_999"):
            build_image_provider(config)

    def test_image_factory_error_raises_provider_init_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ProviderInitError raised when image factory function throws."""
        import llm_gateway.registry as reg

        orig_flag = reg._image_builtins_registered
        reg._image_builtins_registered = True

        def bad_factory(config: GatewayConfig) -> None:
            raise RuntimeError("image factory exploded")

        register_image_provider("broken_image", bad_factory)  # type: ignore[arg-type]

        monkeypatch.setenv("LLM_IMAGE_PROVIDER", "broken_image")
        config = GatewayConfig()
        with pytest.raises(ProviderInitError, match="broken_image"):
            build_image_provider(config)

        # Cleanup
        _IMAGE_PROVIDERS.pop("broken_image", None)
        reg._image_builtins_registered = orig_flag

    def test_list_image_providers(self) -> None:
        """list_image_providers returns a list including fake_image."""
        providers = list_image_providers()
        assert isinstance(providers, list)
        # fake_image is always registered (no optional deps)
        assert "fake_image" in providers

    def test_ensure_image_builtins_registered_runs_once(self) -> None:
        """_ensure_image_builtins_registered is idempotent after first call."""
        import llm_gateway.registry as reg

        orig_flag = reg._image_builtins_registered

        # Force re-registration
        reg._image_builtins_registered = False
        _IMAGE_PROVIDERS.clear()

        reg._ensure_image_builtins_registered()
        first_providers = list(_IMAGE_PROVIDERS.keys())

        # Call again — should be a no-op
        reg._ensure_image_builtins_registered()
        second_providers = list(_IMAGE_PROVIDERS.keys())

        assert first_providers == second_providers
        assert "fake_image" in first_providers
        assert reg._image_builtins_registered is True

        # Restore
        reg._image_builtins_registered = orig_flag

    def test_ensure_image_builtins_registers_fake_image(self) -> None:
        """After fresh registration, fake_image is always present."""
        import llm_gateway.registry as reg

        orig_flag = reg._image_builtins_registered
        saved_providers = dict(_IMAGE_PROVIDERS)

        reg._image_builtins_registered = False
        _IMAGE_PROVIDERS.clear()

        reg._ensure_image_builtins_registered()
        assert "fake_image" in _IMAGE_PROVIDERS

        # Restore
        _IMAGE_PROVIDERS.clear()
        _IMAGE_PROVIDERS.update(saved_providers)
        reg._image_builtins_registered = orig_flag


@pytest.mark.unit
class TestLLMRegistryEdgeCases:
    """Additional edge-case tests for the LLM provider registry."""

    def test_ensure_builtins_registered_is_idempotent(self) -> None:
        """_ensure_builtins_registered short-circuits on second call."""
        import llm_gateway.registry as reg

        # builtins should already be registered from earlier tests
        reg._builtins_registered = True
        saved_providers = dict(_PROVIDERS)

        # Call again — should be a no-op
        reg._ensure_builtins_registered()
        assert dict(_PROVIDERS) == saved_providers

    def test_ensure_builtins_registers_fake(self) -> None:
        """After fresh registration, fake provider is always present."""
        import llm_gateway.registry as reg

        orig_flag = reg._builtins_registered
        saved_providers = dict(_PROVIDERS)

        reg._builtins_registered = False
        _PROVIDERS.clear()

        reg._ensure_builtins_registered()
        assert "fake" in _PROVIDERS

        # Restore
        _PROVIDERS.clear()
        _PROVIDERS.update(saved_providers)
        reg._builtins_registered = orig_flag

    def test_register_provider_overwrites_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Registering the same name twice replaces the factory."""
        factory_a = MagicMock(return_value=MagicMock())
        factory_b = MagicMock(return_value=MagicMock())

        register_provider("overwrite_test", factory_a)
        register_provider("overwrite_test", factory_b)

        monkeypatch.setenv("LLM_PROVIDER", "overwrite_test")
        config = GatewayConfig()
        build_provider(config)

        factory_a.assert_not_called()
        factory_b.assert_called_once_with(config)

        # Cleanup
        _PROVIDERS.pop("overwrite_test", None)

    def test_list_providers_returns_fresh_builtins(self) -> None:
        """list_providers triggers lazy registration and returns builtins."""
        import llm_gateway.registry as reg

        orig_flag = reg._builtins_registered
        saved_providers = dict(_PROVIDERS)

        reg._builtins_registered = False
        _PROVIDERS.clear()

        providers = list_providers()
        assert "fake" in providers
        assert isinstance(providers, list)

        # Restore
        _PROVIDERS.clear()
        _PROVIDERS.update(saved_providers)
        reg._builtins_registered = orig_flag


@pytest.mark.unit
class TestBuiltinImportErrorBranches:
    """Tests that exercise ImportError fallback branches in lazy registration."""

    def test_llm_builtins_handle_anthropic_import_error(self) -> None:
        """When anthropic provider import fails, registration proceeds without it."""
        import llm_gateway.registry as reg

        orig_flag = reg._builtins_registered
        saved_providers = dict(_PROVIDERS)

        reg._builtins_registered = False
        _PROVIDERS.clear()

        # Remove cached module so the import inside _ensure_builtins_registered
        # actually triggers again, then block it.
        saved_modules: dict[str, object] = {}
        mod_name = "llm_gateway.providers.anthropic"
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = builtins.__import__

        def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llm_gateway.providers.anthropic":
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

        try:
            builtins.__import__ = _blocked_import  # type: ignore[assignment]
            reg._ensure_builtins_registered()
        finally:
            builtins.__import__ = original_import
            # Restore cached modules
            for k, v in saved_modules.items():
                sys.modules[k] = v  # type: ignore[assignment]

        # anthropic should NOT be registered, but fake should
        assert "anthropic" not in _PROVIDERS
        assert "fake" in _PROVIDERS

        # Restore
        _PROVIDERS.clear()
        _PROVIDERS.update(saved_providers)
        reg._builtins_registered = orig_flag

    def test_llm_builtins_handle_local_claude_import_error(self) -> None:
        """When local_claude provider import fails, registration proceeds."""
        import llm_gateway.registry as reg

        orig_flag = reg._builtins_registered
        saved_providers = dict(_PROVIDERS)

        reg._builtins_registered = False
        _PROVIDERS.clear()

        saved_modules: dict[str, object] = {}
        mod_name = "llm_gateway.providers.local_claude"
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = builtins.__import__

        def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llm_gateway.providers.local_claude":
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

        try:
            builtins.__import__ = _blocked_import  # type: ignore[assignment]
            reg._ensure_builtins_registered()
        finally:
            builtins.__import__ = original_import
            for k, v in saved_modules.items():
                sys.modules[k] = v  # type: ignore[assignment]

        assert "local_claude" not in _PROVIDERS
        assert "fake" in _PROVIDERS

        # Restore
        _PROVIDERS.clear()
        _PROVIDERS.update(saved_providers)
        reg._builtins_registered = orig_flag

    def test_llm_builtins_handle_gemini_import_error(self) -> None:
        """When gemini provider import fails, registration proceeds."""
        import llm_gateway.registry as reg

        orig_flag = reg._builtins_registered
        saved_providers = dict(_PROVIDERS)

        reg._builtins_registered = False
        _PROVIDERS.clear()

        saved_modules: dict[str, object] = {}
        mod_name = "llm_gateway.providers.gemini"
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = builtins.__import__

        def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llm_gateway.providers.gemini":
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

        try:
            builtins.__import__ = _blocked_import  # type: ignore[assignment]
            reg._ensure_builtins_registered()
        finally:
            builtins.__import__ = original_import
            for k, v in saved_modules.items():
                sys.modules[k] = v  # type: ignore[assignment]

        assert "gemini" not in _PROVIDERS
        assert "fake" in _PROVIDERS

        # Restore
        _PROVIDERS.clear()
        _PROVIDERS.update(saved_providers)
        reg._builtins_registered = orig_flag

    def test_image_builtins_handle_openai_image_import_error(self) -> None:
        """When openai_image provider import fails, registration proceeds."""
        import llm_gateway.registry as reg

        orig_flag = reg._image_builtins_registered
        saved_providers = dict(_IMAGE_PROVIDERS)

        reg._image_builtins_registered = False
        _IMAGE_PROVIDERS.clear()

        saved_modules: dict[str, object] = {}
        mod_name = "llm_gateway.providers.openai_image"
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = builtins.__import__

        def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llm_gateway.providers.openai_image":
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

        try:
            builtins.__import__ = _blocked_import  # type: ignore[assignment]
            reg._ensure_image_builtins_registered()
        finally:
            builtins.__import__ = original_import
            for k, v in saved_modules.items():
                sys.modules[k] = v  # type: ignore[assignment]

        assert "openai_image" not in _IMAGE_PROVIDERS
        assert "fake_image" in _IMAGE_PROVIDERS

        # Restore
        _IMAGE_PROVIDERS.clear()
        _IMAGE_PROVIDERS.update(saved_providers)
        reg._image_builtins_registered = orig_flag
