#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration for Forgather tests.
"""

import pytest
import tempfile
import shutil
import torch
import torch.nn as nn
from unittest.mock import Mock

from forgather.ml.trainer_types import TrainingArguments


class SimpleMockModel(nn.Module):
    """Simple model for testing purposes"""
    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def training_args(temp_dir):
    """Default training arguments for tests"""
    return TrainingArguments(
        output_dir=temp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=3
    )


@pytest.fixture
def mock_model():
    """Simple mock model for testing"""
    return SimpleMockModel()


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing"""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=10)
    return dataset


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing"""
    def _create_optimizer(model_params):
        return torch.optim.Adam(model_params, lr=0.001)
    return _create_optimizer


@pytest.fixture
def mock_scheduler():
    """Mock scheduler factory for testing"""
    def _create_scheduler(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    return _create_scheduler