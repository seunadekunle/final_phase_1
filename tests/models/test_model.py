import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import memory_profiler

from src.models.clip_encoder import CLIPEncoder
from src.models.style_classifier import StyleClassifier

@pytest.fixture
def sample_batch():
    """Create sample batch of images."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    category_labels = torch.randint(0, 10, (batch_size,))
    attribute_labels = torch.randint(0, 2, (batch_size, 20)).float()
    return images, category_labels, attribute_labels

def test_clip_encoder_initialization():
    """Test CLIP encoder initialization."""
    encoder = CLIPEncoder()
    assert encoder.model is not None
    assert encoder.processor is not None
    
    # Check if parameters are frozen
    for param in encoder.model.parameters():
        assert not param.requires_grad

def test_clip_encoder_forward(sample_batch):
    """Test CLIP encoder forward pass."""
    encoder = CLIPEncoder()
    images = sample_batch[0]
    
    features = encoder(images)
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == images.shape[0]
    assert features.shape[1] == encoder.model.config.projection_dim

@pytest.mark.memory
def test_clip_encoder_memory(sample_batch):
    """Test CLIP encoder memory usage."""
    def measure_clip():
        encoder = CLIPEncoder()
        features = encoder(sample_batch[0])
        del encoder, features
        torch.cuda.empty_cache()
        
    mem_usage = memory_profiler.memory_usage((measure_clip, (), {}))
    peak_mem = max(mem_usage) - min(mem_usage)
    
    # Memory usage should be less than 2GB
    assert peak_mem < 2000

def test_style_classifier_initialization():
    """Test style classifier initialization."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20,
        hidden_dim=512,
        dropout=0.1
    )
    
    assert isinstance(model.encoder, CLIPEncoder)
    assert model.category_classifier[-1].out_features == 10
    assert model.attribute_classifier[-1].out_features == 20

def test_style_classifier_forward(sample_batch):
    """Test style classifier forward pass."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    images = sample_batch[0]
    category_logits, attribute_logits = model(images)
    
    assert category_logits.shape == (images.shape[0], 10)
    assert attribute_logits.shape == (images.shape[0], 20)

def test_style_classifier_loss(sample_batch):
    """Test style classifier loss computation."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    images, category_labels, attribute_labels = sample_batch
    category_logits, attribute_logits = model(images)
    
    loss_dict = model.compute_loss(
        category_logits,
        attribute_logits,
        category_labels,
        attribute_labels
    )
    
    assert "total_loss" in loss_dict
    assert "category_loss" in loss_dict
    assert "attribute_loss" in loss_dict
    assert all(isinstance(v, torch.Tensor) for v in loss_dict.values())

def test_style_classifier_predict(sample_batch):
    """Test style classifier prediction."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    images = sample_batch[0]
    category_preds, attribute_preds = model.predict(images)
    
    assert category_preds.shape == (images.shape[0],)
    assert attribute_preds.shape == (images.shape[0], 20)
    assert torch.all((attribute_preds >= 0) & (attribute_preds <= 1))

@pytest.mark.memory
def test_style_classifier_memory(sample_batch):
    """Test style classifier memory usage."""
    def measure_model():
        model = StyleClassifier(
            num_categories=10,
            num_attributes=20
        )
        category_preds, attribute_preds = model.predict(sample_batch[0])
        del model, category_preds, attribute_preds
        torch.cuda.empty_cache()
        
    mem_usage = memory_profiler.memory_usage((measure_model, (), {}))
    peak_mem = max(mem_usage) - min(mem_usage)
    
    # Memory usage should be less than 4GB
    assert peak_mem < 4000

@pytest.mark.benchmark
def test_model_inference_speed(sample_batch, benchmark):
    """Test model inference speed."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    def run_inference():
        with torch.no_grad():
            _ = model.predict(sample_batch[0])
            
    # Should process batch in less than 100ms on GPU
    benchmark(run_inference)

def test_model_device_placement():
    """Test model device placement."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    # Test MPS device if available
    if torch.backends.mps.is_available():
        model = model.to("mps")
        assert next(model.parameters()).device.type == "mps"
    
    # Test CPU
    model = model.to("cpu")
    assert next(model.parameters()).device.type == "cpu"

def test_gradient_flow(sample_batch):
    """Test gradient flow through the model."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    images, category_labels, attribute_labels = sample_batch
    category_logits, attribute_logits = model(images)
    
    loss_dict = model.compute_loss(
        category_logits,
        attribute_logits,
        category_labels,
        attribute_labels
    )
    
    loss_dict["total_loss"].backward()
    
    # Check if gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            
def test_model_save_load(sample_batch, tmp_path):
    """Test model saving and loading."""
    model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    
    # Save model
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    loaded_model = StyleClassifier(
        num_categories=10,
        num_attributes=20
    )
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Compare outputs
    with torch.no_grad():
        out1 = model(sample_batch[0])
        out2 = loaded_model(sample_batch[0])
        
    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out1[1], out2[1]) 