"""
Base generation strategy abstract class.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Union

if TYPE_CHECKING:
    from ..service import InferenceService


class GenerationStrategy(ABC):
    """
    Abstract base class for generation strategies.

    All generation strategies must inherit from this class and implement
    the generate() method. Strategies receive an InferenceService instance
    which provides access to the model, tokenizer, and utility functions.

    Example:
        Creating a custom strategy:
        >>> class MyCustomStrategy(GenerationStrategy):
        ...     def generate(self, request):
        ...         # Use self.service.model, self.service.tokenizer, etc.
        ...         prompt = self.service.format_messages(request.messages)
        ...         # ... perform generation ...
        ...         return response
        ...
        >>> strategy = MyCustomStrategy(service)
        >>> response = strategy.generate(request)

    Attributes:
        service: InferenceService instance with model and utilities
    """

    def __init__(self, service: "InferenceService") -> None:
        """
        Initialize strategy with service reference.

        Args:
            service: InferenceService instance providing model, tokenizer, etc.
        """
        self.service = service

    @abstractmethod
    def generate(self, request) -> Union[object, Iterator[str]]:
        """
        Generate response for the given request.

        This method must be implemented by all strategy subclasses.

        Args:
            request: Request object (ChatCompletionRequest or CompletionRequest)

        Returns:
            Response object for non-streaming, Iterator[str] for streaming

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
