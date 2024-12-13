import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
import time
import psutil
import logging
from PIL import Image

from src.configs.config import Config
from src.data.datamodule import DeepFashionDataModule
from src.models.style_classifier import StyleClassifier
from legacy.style_classifier.src.models.trainer import StyleTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data/DeepFashion"
    os.makedirs(data_dir / "img")
    
    # Create sample images
    for i in range(10):
        img = Image.new('RGB', (224, 224), color=f'#{i:06x}')
        img.save(data_dir / f"img/image_{i}.jpg")
    
    # Create annotation files
    splits = ['train', 'val', 'test']
    num_images = {'train': 6, 'val': 2, 'test': 2}
    
    for split in splits:
        # Split file
        with open(data_dir / f"{split}.txt", "w") as f:
            f.write(f"{num_images[split]}\n")
            f.write("image_path\n")
            start_idx = 0 if split == 'train' else (6 if split == 'val' else 8)
            for i in range(start_idx, start_idx + num_images[split]):
                f.write(f"img/image_{i}.jpg\n")
                
        # Category file
        with open(data_dir / f"{split}_cate.txt", "w") as f:
            f.write(f"{num_images[split]}\n")
            f.write("image_path category_id\n")
            for i in range(start_idx, start_idx + num_images[split]):
                f.write(f"img/image_{i}.jpg {i % 3 + 1}\n")
                
        # Attribute file
        with open(data_dir / f"{split}_attr.txt", "w") as f:
            f.write(f"{num_images[split]}\n")
            f.write("image_path attribute_labels\n")
            for i in range(start_idx, start_idx + num_images[split]):
                attrs = [str(i % 2) for _ in range(5)]
                f.write(f"img/image_{i}.jpg {' '.join(attrs)}\n")
                
        # Bbox file
        with open(data_dir / f"{split}_bbox.txt", "w") as f:
            f.write(f"{num_images[split]}\n")
            f.write("image_path x1 y1 x2 y2\n")
            for i in range(start_idx, start_idx + num_images[split]):
                f.write(f"img/image_{i}.jpg 10 10 200 200\n")
    
    # Category list
    with open(data_dir / "list_category_cloth.txt", "w") as f:
        f.write("3\n")
        f.write("category_id category_name\n")
        f.write("1 T-shirt\n")
        f.write("2 Dress\n")
        f.write("3 Pants\n")
        
    # Attribute list
    with open(data_dir / "list_attr_cloth.txt", "w") as f:
        f.write("5\n")
        f.write("attribute_name\n")
        f.write("Floral\n")
        f.write("Striped\n")
        f.write("Casual\n")
        f.write("Formal\n")
        f.write("Vintage\n")
        
    yield data_dir
    shutil.rmtree(temp_dir)

def test_end_to_end_training(sample_dataset):
    """Test complete training pipeline."""
    # Create config
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    config.training.epochs = 2
    
    # Initialize data module
    data_module = DeepFashionDataModule(config.data)
    data_module.setup()
    
    # Create model
    model = StyleClassifier(
        num_categories=3,
        num_attributes=5,
        hidden_dim=256,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = StyleTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    # Train model
    metrics = trainer.train()
    
    # Verify metrics
    assert "loss" in metrics
    assert "category_accuracy" in metrics
    assert "attribute_f1" in metrics
    
    # Test predictions
    test_metrics = trainer.test()
    assert all(k in test_metrics for k in ["loss", "category_accuracy", "attribute_f1"])

def test_memory_usage(sample_dataset):
    """Test memory usage during training."""
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    initial_memory = get_memory_usage()
    
    # Create config and data module
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    config.training.epochs = 1
    
    data_module = DeepFashionDataModule(config.data)
    data_module.setup()
    
    # Create model and trainer
    model = StyleClassifier(
        num_categories=3,
        num_attributes=5,
        hidden_dim=256
    )
    
    trainer = StyleTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    # Train for one epoch
    trainer.train()
    
    peak_memory = get_memory_usage()
    memory_increase = peak_memory - initial_memory
    
    logger.info(f"Memory usage: {memory_increase:.2f} MB")
    assert memory_increase < 4000  # Less than 4GB increase

def test_training_speed(sample_dataset):
    """Test training speed."""
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    config.training.epochs = 1
    
    data_module = DeepFashionDataModule(config.data)
    data_module.setup()
    
    model = StyleClassifier(
        num_categories=3,
        num_attributes=5,
        hidden_dim=256
    )
    
    trainer = StyleTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training time: {training_time:.2f}s")
    assert training_time < 60  # Should complete within 60 seconds

def test_checkpoint_saving(sample_dataset, tmp_path):
    """Test model checkpoint saving and loading."""
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    config.training.epochs = 1
    
    data_module = DeepFashionDataModule(config.data)
    data_module.setup()
    
    model = StyleClassifier(
        num_categories=3,
        num_attributes=5,
        hidden_dim=256
    )
    
    trainer = StyleTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    # Train and save checkpoint
    trainer.train()
    assert trainer.best_model_path is not None
    assert trainer.best_model_path.exists()
    
    # Load checkpoint
    new_trainer = StyleTrainer(
        model=StyleClassifier(
            num_categories=3,
            num_attributes=5,
            hidden_dim=256
        ),
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    new_trainer.load_checkpoint(trainer.best_model_path)
    
    # Compare predictions
    test_batch = next(iter(data_module.test_dataloader()))
    with torch.no_grad():
        pred1 = trainer.model(test_batch[0])
        pred2 = new_trainer.model(test_batch[0])
        
    assert torch.allclose(pred1[0], pred2[0])
    assert torch.allclose(pred1[1], pred2[1])

def test_error_handling(sample_dataset):
    """Test error handling in the pipeline."""
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    
    # Test invalid category number
    with pytest.raises(Exception):
        model = StyleClassifier(
            num_categories=0,  # Invalid
            num_attributes=5
        )
    
    # Test invalid attribute number
    with pytest.raises(Exception):
        model = StyleClassifier(
            num_categories=3,
            num_attributes=-1  # Invalid
        )
    
    # Test invalid data directory
    config.data.data_dir = Path("nonexistent")
    with pytest.raises(Exception):
        data_module = DeepFashionDataModule(config.data)
        data_module.setup()

def test_device_compatibility(sample_dataset):
    """Test device compatibility."""
    config = Config()
    config.data.data_dir = sample_dataset
    config.data.batch_size = 2
    config.training.epochs = 1
    
    # Test MPS device if available
    if torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"
    
    data_module = DeepFashionDataModule(config.data)
    data_module.setup()
    
    model = StyleClassifier(
        num_categories=3,
        num_attributes=5
    )
    
    trainer = StyleTrainer(
        model=model,
        config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    # Verify device placement
    assert next(trainer.model.parameters()).device.type == config.device
    
    # Train for one epoch
    trainer.train() 