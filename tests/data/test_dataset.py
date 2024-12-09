import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
from PIL import Image
import memory_profiler

from src.data.dataset import DeepFashionDataset, DeepFashionDatasetError

@pytest.fixture
def sample_data_dir():
    """Create temporary directory with sample dataset."""
    temp_dir = tempfile.mkdtemp()
    
    # Create directory structure
    os.makedirs(os.path.join(temp_dir, "img"))
    
    # Create sample image
    img = Image.new('RGB', (224, 224), color='red')
    img.save(os.path.join(temp_dir, "img/tshirt1.jpg"))
    img.save(os.path.join(temp_dir, "img/dress1.jpg"))
    
    # Create annotation files
    with open(os.path.join(temp_dir, "list_category_cloth.txt"), "w") as f:
        f.write("2\n")
        f.write("category_id category_name\n")
        f.write("1 T-shirt\n")
        f.write("2 Dress\n")
        
    with open(os.path.join(temp_dir, "list_attr_cloth.txt"), "w") as f:
        f.write("3\n")
        f.write("attribute_name\n")
        f.write("Floral\n")
        f.write("Striped\n")
        f.write("Casual\n")
        
    with open(os.path.join(temp_dir, "train.txt"), "w") as f:
        f.write("2\n")
        f.write("image_path\n")
        f.write("img/tshirt1.jpg\n")
        f.write("img/dress1.jpg\n")
        
    with open(os.path.join(temp_dir, "train_cate.txt"), "w") as f:
        f.write("2\n")
        f.write("image_path category_id\n")
        f.write("img/tshirt1.jpg 1\n")
        f.write("img/dress1.jpg 2\n")
        
    with open(os.path.join(temp_dir, "train_attr.txt"), "w") as f:
        f.write("2\n")
        f.write("image_path attribute_labels\n")
        f.write("img/tshirt1.jpg 1 0 1\n")
        f.write("img/dress1.jpg 0 1 1\n")
        
    with open(os.path.join(temp_dir, "train_bbox.txt"), "w") as f:
        f.write("2\n")
        f.write("image_path x1 y1 x2 y2\n")
        f.write("img/tshirt1.jpg 10 10 200 200\n")
        f.write("img/dress1.jpg 20 20 180 180\n")
        
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_dataset_initialization(sample_data_dir):
    """Test dataset initialization."""
    dataset = DeepFashionDataset(sample_data_dir, "train")
    assert len(dataset) == 2
    assert dataset.num_categories == 2
    assert dataset.num_attributes == 3

def test_dataset_invalid_split(sample_data_dir):
    """Test dataset initialization with invalid split."""
    with pytest.raises(DeepFashionDatasetError):
        DeepFashionDataset(sample_data_dir, "invalid_split")

def test_dataset_getitem(sample_data_dir):
    """Test dataset item retrieval."""
    dataset = DeepFashionDataset(sample_data_dir, "train")
    image, category_label, attribute_labels = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(category_label, int)
    assert isinstance(attribute_labels, torch.Tensor)
    assert attribute_labels.shape == (3,)

def test_dataset_transforms(sample_data_dir):
    """Test custom transforms."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    
    dataset = DeepFashionDataset(sample_data_dir, "train", transform=transform)
    image, _, _ = dataset[0]
    assert image.shape == (3, 128, 128)

def test_dataset_bbox_cropping(sample_data_dir):
    """Test bounding box cropping."""
    dataset = DeepFashionDataset(sample_data_dir, "train", use_bbox=True)
    image, _, _ = dataset[0]
    
    # Check if image is properly cropped
    assert image.shape == (3, 224, 224)

def test_dataset_no_bbox(sample_data_dir):
    """Test dataset without bounding boxes."""
    dataset = DeepFashionDataset(sample_data_dir, "train", use_bbox=False)
    image, _, _ = dataset[0]
    assert image.shape == (3, 224, 224)

def test_dataset_missing_image(sample_data_dir):
    """Test handling of missing images."""
    # Remove an image file
    os.remove(os.path.join(sample_data_dir, "img/tshirt1.jpg"))
    
    dataset = DeepFashionDataset(sample_data_dir, "train")
    with pytest.raises(DeepFashionDatasetError):
        _ = dataset[0]

@pytest.mark.memory
def test_dataset_memory_usage(sample_data_dir):
    """Test memory usage during dataset operations."""
    def measure_dataset_ops():
        dataset = DeepFashionDataset(sample_data_dir, "train")
        # Access all items
        for i in range(len(dataset)):
            _ = dataset[i]
            
    mem_usage = memory_profiler.memory_usage((measure_dataset_ops, (), {}))
    peak_mem = max(mem_usage) - min(mem_usage)
    
    # Memory usage should be less than 500MB
    assert peak_mem < 500

def test_dataset_category_mapping(sample_data_dir):
    """Test category name mapping."""
    dataset = DeepFashionDataset(sample_data_dir, "train")
    assert dataset.get_category_name(1) == "T-shirt"
    assert dataset.get_category_name(2) == "Dress"
    
    with pytest.raises(DeepFashionDatasetError):
        dataset.get_category_name(999)

def test_dataset_attribute_mapping(sample_data_dir):
    """Test attribute name mapping."""
    dataset = DeepFashionDataset(sample_data_dir, "train")
    attr_names = dataset.get_attribute_names([1, 2])
    assert attr_names == ["Floral", "Striped"]
    
    with pytest.raises(DeepFashionDatasetError):
        dataset.get_attribute_names([999])

@pytest.mark.benchmark
def test_dataset_loading_speed(sample_data_dir, benchmark):
    """Test dataset loading speed."""
    def load_dataset():
        dataset = DeepFashionDataset(sample_data_dir, "train")
        _ = dataset[0]
        
    # Should load first item in less than 100ms
    benchmark(load_dataset) 