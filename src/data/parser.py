"""parser module for deepfashion dataset annotations and splits"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DeepFashionParserError(Exception):
    """custom exception for deepfashion parser errors"""
    pass

class DeepFashionParser:
    """parser for deepfashion dataset files and annotations
    
    args:
        data_dir: path to dataset directory
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        if not self.data_dir.exists():
            raise DeepFashionParserError(f"data directory {data_dir} does not exist")
            
    def _check_image_exists(self, image_path: str) -> bool:
        """check if image exists in dataset
        
        args:
            image_path: path to image relative to data directory
            
        returns:
            bool: True if image exists, False otherwise
        """
        # normalize path separators
        image_path = Path(image_path).as_posix()
        
        # handle paths with or without img/ prefix
        if not image_path.startswith('img/'):
            image_path = f"img/{image_path}"
            
        # split path into components
        parts = image_path.split('/')
        if len(parts) != 3:  # should be ['img', 'category_dir', 'filename.jpg']
            logger.warning(f"invalid image path format: {image_path}")
            return False
            
        # construct full path
        full_path = self.data_dir / parts[0] / parts[1] / parts[2]
        exists = full_path.exists()
        
        if not exists:
            logger.warning(f"image not found: {full_path}")
            
        return exists
            
    def parse_category_list(self, filename: str) -> Tuple[Dict[int, str], int]:
        """parse category list file
        
        args:
            filename: name of category list file
            
        returns:
            tuple: (category_dict, num_categories)
            - category_dict: mapping from category id to name
            - num_categories: total number of categories
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            categories = {}
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                # first line contains number of categories
                num_categories = int(lines[0].strip())
                
                # skip header line
                current_id = 1  # start category ids at 1
                
                # parse remaining lines
                for line in lines[2:]:  # skip count and header
                    category_name = line.strip().split()[0]  # get first column
                    if category_name:  # skip empty lines
                        categories[current_id] = category_name
                        current_id += 1
                        
            if len(categories) != num_categories:
                raise DeepFashionParserError(
                    f"mismatch in category count: expected {num_categories}, found {len(categories)}"
                )
                    
            return categories, num_categories
            
        except (FileNotFoundError, ValueError) as e:
            raise DeepFashionParserError(f"failed to parse category file: {str(e)}")
            
    def parse_attribute_list(self, filename: str) -> Tuple[Dict[int, str], int]:
        """parse attribute list file
        
        args:
            filename: name of attribute list file
            
        returns:
            tuple: (attribute_dict, num_attributes)
            - attribute_dict: mapping from attribute id to name
            - num_attributes: total number of attributes
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            attributes = {}
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                # first line contains number of attributes
                num_attributes = int(lines[0].strip())
                
                # skip header line
                current_id = 1  # start attribute ids at 1
                
                # parse remaining lines
                for line in lines[2:]:  # skip count and header
                    attr_name = line.strip().split()[0]  # get first column
                    if attr_name:  # skip empty lines
                        attributes[current_id] = attr_name
                        current_id += 1
                        
            if len(attributes) != num_attributes:
                raise DeepFashionParserError(
                    f"mismatch in attribute count: expected {num_attributes}, found {len(attributes)}"
                )
                    
            return attributes, num_attributes
            
        except (FileNotFoundError, ValueError) as e:
            raise DeepFashionParserError(f"failed to parse attribute file: {str(e)}")
            
    def parse_split_file(self, filename: str) -> List[str]:
        """parse split file containing image paths
        
        args:
            filename: name of split file
            
        returns:
            list: list of image paths
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            image_paths = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path:  # skip empty lines
                        # normalize path
                        path = Path(path).as_posix()
                        image_paths.append(path)
                        
            if not image_paths:
                raise DeepFashionParserError("no valid images found in split file")
                
            return image_paths
            
        except FileNotFoundError as e:
            raise DeepFashionParserError(f"failed to parse split file: {str(e)}")
            
    def parse_category_file(self, filename: str) -> List[int]:
        """parse category file containing category labels
        
        args:
            filename: name of category file
            
        returns:
            list: list of category labels
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            categories = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    category = line.strip()
                    if category:  # skip empty lines
                        categories.append(int(category))
                        
            return categories
            
        except (FileNotFoundError, ValueError) as e:
            raise DeepFashionParserError(f"failed to parse category file: {str(e)}")
            
    def parse_attribute_file(self, filename: str, num_attributes: int) -> List[List[int]]:
        """parse attribute file containing attribute labels
        
        args:
            filename: name of attribute file
            num_attributes: total number of attributes
            
        returns:
            list: list of attribute label lists
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            attributes = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    attrs = line.strip().split()
                    if attrs:  # skip empty lines
                        if len(attrs) != num_attributes:
                            raise DeepFashionParserError(
                                f"mismatch in attribute count: expected {num_attributes}, found {len(attrs)}"
                            )
                        attributes.append([int(a) for a in attrs])
                        
            return attributes
            
        except (FileNotFoundError, ValueError) as e:
            raise DeepFashionParserError(f"failed to parse attribute file: {str(e)}")
            
    def parse_bbox_file(self, filename: str) -> List[List[int]]:
        """parse bbox file containing bounding box coordinates
        
        args:
            filename: name of bbox file
            
        returns:
            list: list of bounding box coordinate lists [x1, y1, x2, y2]
            
        raises:
            DeepFashionParserError: if file cannot be parsed
        """
        try:
            # handle absolute paths
            if Path(filename).is_absolute():
                filepath = Path(filename)
            else:
                filepath = self.data_dir / filename
                
            bboxes = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    coords = line.strip().split()
                    if coords:  # skip empty lines
                        if len(coords) != 4:
                            raise DeepFashionParserError(
                                f"invalid bbox format: expected 4 coordinates, found {len(coords)}"
                            )
                        bboxes.append([int(c) for c in coords])
                        
            return bboxes
            
        except (FileNotFoundError, ValueError) as e:
            raise DeepFashionParserError(f"failed to parse bbox file: {str(e)}")