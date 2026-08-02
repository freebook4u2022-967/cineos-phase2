"""Audio provider registry."""

from .provider import FakeDeterministicProvider, TextToSpeechProvider


class ProviderRegistry:
    def __init__(self, *, include_fake: bool = True) -> None:
        self._providers: dict[str, TextToSpeechProvider] = {}
        if include_fake:
            self.register(FakeDeterministicProvider())

    def register(self, provider: TextToSpeechProvider) -> None:
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> TextToSpeechProvider:
        try:
            return self._providers[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown audio provider: {provider_id}") from error

    def list(self) -> list[TextToSpeechProvider]:
        return [self._providers[key] for key in sorted(self._providers)]
