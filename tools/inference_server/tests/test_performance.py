"""
Performance profiling tests to verify no regression from refactoring.
"""

import pytest
import time
import torch
from unittest.mock import Mock, patch
from ..service import InferenceService
from ..strategies import ChatGenerationStrategy, CompletionGenerationStrategy
from ..models.chat import ChatMessage, ChatCompletionRequest
from ..models.completion import CompletionRequest


class TestPerformance:
    """Performance tests to ensure refactoring didn't introduce overhead."""

    @pytest.fixture
    def mock_service_components(self):
        """Create mocked service components for performance testing."""
        with (
            patch("..service.AutoTokenizer"),
            patch("..service.AutoModelForCausalLM"),
            patch("..service.GenerationConfig"),
        ):

            # Create service with minimal overhead
            service = Mock()

            # Mock tokenizer
            service.tokenizer = Mock()
            service.tokenizer.decode = Mock(return_value="response")
            service.tokenizer.eos_token_id = 2
            service.tokenizer.bos_token = "<s>"
            service.tokenizer.eos_token = "</s>"

            # Mock model
            service.model = Mock()
            mock_output = Mock()
            mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
            service.model.generate = Mock(return_value=mock_output)

            # Mock utilities
            service.tokenizer_wrapper = Mock()
            service.tokenizer_wrapper.tokenize_and_move_to_device = Mock(
                return_value={
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "prompt_tokens": 5,
                }
            )

            service.stop_processor = Mock()
            service.stop_processor.process = Mock(
                return_value=([6, 7, 8], torch.tensor([6, 7, 8]), False, None)
            )

            service.finish_detector = Mock()
            service.finish_detector.determine_finish_reason = Mock(return_value="stop")

            service.logger = Mock()
            service.logger.log_request = Mock()
            service.logger.log_messages = Mock()
            service.logger.log_prompt = Mock()
            service.logger.log_input_tokens = Mock()
            service.logger.log_generation_config = Mock()
            service.logger.log_stop_strings = Mock()
            service.logger.log_generated_tokens = Mock()
            service.logger.log_response = Mock()

            service.stop_sequences = []
            service.format_messages = Mock(return_value="formatted prompt")
            service._build_generation_config = Mock(return_value=Mock())

            return service

    def test_chat_generation_performance(self, mock_service_components, benchmark=None):
        """Benchmark chat generation strategy."""
        strategy = ChatGenerationStrategy(mock_service_components)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
        )

        # Warmup
        for _ in range(3):
            strategy.generate(request)

        # Benchmark
        times = []
        iterations = 100

        for _ in range(iterations):
            start = time.perf_counter()
            strategy.generate(request)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nChat Generation Performance:")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Min: {min_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")

        # Assert reasonable performance (should be fast with mocks)
        assert avg_time < 0.001, f"Average time {avg_time}s exceeds 1ms threshold"

    def test_completion_generation_performance(self, mock_service_components):
        """Benchmark completion generation strategy."""
        strategy = CompletionGenerationStrategy(mock_service_components)

        request = CompletionRequest(
            model="test-model", prompt="Test prompt", max_tokens=50
        )

        # Warmup
        for _ in range(3):
            strategy.generate(request)

        # Benchmark
        times = []
        iterations = 100

        for _ in range(iterations):
            start = time.perf_counter()
            strategy.generate(request)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nCompletion Generation Performance:")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Min: {min_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")

        # Assert reasonable performance
        assert avg_time < 0.001, f"Average time {avg_time}s exceeds 1ms threshold"

    def test_stop_processor_performance(self, mock_service_components):
        """Benchmark stop sequence processing."""
        from ..core.stop_processor import StopSequenceProcessor

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        processor = StopSequenceProcessor(tokenizer)

        text = "Generated text STOP more content"
        token_ids = [1, 2, 3, 4, 5, 6]
        tokens = torch.tensor(token_ids)
        stop_sequences = ["STOP", "END", "FINISH"]

        # Warmup
        for _ in range(10):
            processor.process(text, token_ids, tokens, stop_sequences)

        # Benchmark
        times = []
        iterations = 1000

        for _ in range(iterations):
            start = time.perf_counter()
            processor.process(text, token_ids, tokens, stop_sequences)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)

        print(f"\nStop Processor Performance:")
        print(f"  Average: {avg_time*1000000:.1f}μs")

        # Should be very fast (< 100 microseconds)
        assert avg_time < 0.0001, f"Stop processing too slow: {avg_time}s"

    def test_finish_detector_performance(self, mock_service_components):
        """Benchmark finish reason detection."""
        from ..core.finish_detector import FinishReasonDetector

        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        detector = FinishReasonDetector(tokenizer, {2, 50256})

        token_ids = [1, 3, 4, 5, 6]
        max_tokens = 10
        stopped_by_sequence = False

        # Warmup
        for _ in range(10):
            detector.determine_finish_reason(token_ids, max_tokens, stopped_by_sequence)

        # Benchmark
        times = []
        iterations = 10000

        for _ in range(iterations):
            start = time.perf_counter()
            detector.determine_finish_reason(token_ids, max_tokens, stopped_by_sequence)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)

        print(f"\nFinish Detector Performance:")
        print(f"  Average: {avg_time*1000000:.1f}μs")

        # Should be extremely fast (< 10 microseconds)
        assert avg_time < 0.00001, f"Finish detection too slow: {avg_time}s"

    def test_memory_overhead(self, mock_service_components):
        """Test that strategy pattern doesn't add significant memory overhead."""
        import sys

        # Measure strategy size
        strategy = ChatGenerationStrategy(mock_service_components)

        strategy_size = sys.getsizeof(strategy)
        service_ref_size = sys.getsizeof(strategy.service)

        print(f"\nMemory Overhead:")
        print(f"  Strategy object: {strategy_size} bytes")
        print(f"  Service reference: {service_ref_size} bytes")

        # Strategy should be lightweight (just holds a reference)
        assert strategy_size < 1000, f"Strategy too large: {strategy_size} bytes"

    def test_concurrent_requests_performance(self, mock_service_components):
        """Test performance with multiple concurrent-style requests."""
        strategy = ChatGenerationStrategy(mock_service_components)

        requests = [
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content=f"Message {i}")],
                max_tokens=100,
            )
            for i in range(10)
        ]

        # Benchmark sequential processing
        start = time.perf_counter()
        for request in requests:
            strategy.generate(request)
        end = time.perf_counter()

        total_time = end - start
        avg_per_request = total_time / len(requests)

        print(f"\nConcurrent-style Requests:")
        print(f"  Total time for 10 requests: {total_time*1000:.1f}ms")
        print(f"  Average per request: {avg_per_request*1000:.3f}ms")

        # With mocks, should be very fast
        assert avg_per_request < 0.001, f"Per-request time too slow: {avg_per_request}s"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available for GPU performance tests"
)
class TestGPUPerformance:
    """GPU-specific performance tests (skipped if CUDA unavailable)."""

    def test_device_placement_overhead(self):
        """Test that device placement doesn't add significant overhead."""
        # This would test actual device placement with real tensors
        # Skipped if no GPU available
        pass


if __name__ == "__main__":
    # Allow running performance tests directly
    pytest.main([__file__, "-v", "-s"])
