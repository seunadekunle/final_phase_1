import pytest
from pathlib import Path
import tempfile
import shutil
import os
import memory_profiler

from src.data.parser import DeepFashionParser, DeepFashionParserError

@pytest.fixture
def sample_data_dir():
    """Create temporary directory with sample data files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample category file
    with open(os.path.join(temp_dir, "list_category_cloth.txt"), "w") as f:
        f.write("2\n")
        f.write("category_id category_name\n")
        f.write("1 T-shirt\n")
        f.write("2 Dress\n")
        
    # Create sample attribute file
    with open(os.path.join(temp_dir, "list_attr_cloth.txt"), "w") as f:
        f.write("3\n")
        f.write("attribute_name\n")
        f.write("Floral\n")
        f.write("Striped\n")
        f.write("Casual\n")
        
    # Create sample split file
    with open(os.path.join(temp_dir, "train.txt"), "w") as f:
        f.write("2\n")
        f.write("image_path\n")
        f.write("img/tshirt1.jpg\n")
        f.write("img/dress1.jpg\n")
        
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_parser_initialization(sample_data_dir):
    """Test parser initialization."""
    parser = DeepFashionParser(sample_data_dir)
    assert parser.data_dir == sample_data_dir

def test_parser_invalid_directory():
    """Test parser initialization with invalid directory."""
    with pytest.raises(DeepFashionParserError):
        DeepFashionParser(Path("nonexistent_dir"))

def test_parse_category_list(sample_data_dir):
    """Test category list parsing."""
    parser = DeepFashionParser(sample_data_dir)
    categories, num_categories = parser.parse_category_list("list_category_cloth.txt")
    
    assert num_categories == 2
    assert len(categories) == 2
    assert categories[1] == "T-shirt"
    assert categories[2] == "Dress"

def test_parse_attribute_list(sample_data_dir):
    """Test attribute list parsing."""
    parser = DeepFashionParser(sample_data_dir)
    attributes, num_attributes = parser.parse_attribute_list("list_attr_cloth.txt")
    
    assert num_attributes == 3
    assert len(attributes) == 3
    assert attributes[1] == "Floral"
    assert attributes[2] == "Striped"
    assert attributes[3] == "Casual"

def test_parse_split(sample_data_dir):
    """Test split file parsing."""
    parser = DeepFashionParser(sample_data_dir)
    image_paths = parser.parse_split("train.txt")
    
    assert len(image_paths) == 2
    assert image_paths[0] == "img/tshirt1.jpg"
    assert image_paths[1] == "img/dress1.jpg"

def test_invalid_file(sample_data_dir):
    """Test parsing invalid file."""
    parser = DeepFashionParser(sample_data_dir)
    with pytest.raises(DeepFashionParserError):
        parser.parse_category_list("nonexistent_file.txt")

def test_malformed_category_file(sample_data_dir):
    """Test parsing malformed category file."""
    with open(os.path.join(sample_data_dir, "malformed.txt"), "w") as f:
        f.write("invalid\n")
        
    parser = DeepFashionParser(sample_data_dir)
    with pytest.raises(DeepFashionParserError):
        parser.parse_category_list("malformed.txt")

@pytest.mark.memory
def test_memory_usage(sample_data_dir):
    """Test memory usage during parsing."""
    def measure_parsing():
        parser = DeepFashionParser(sample_data_dir)
        parser.parse_category_list("list_category_cloth.txt")
        parser.parse_attribute_list("list_attr_cloth.txt")
        parser.parse_split("train.txt")
        
    mem_usage = memory_profiler.memory_usage((measure_parsing, (), {}))
    peak_mem = max(mem_usage) - min(mem_usage)
    
    # Memory usage should be less than 100MB
    assert peak_mem < 100 