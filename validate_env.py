#!/usr/bin/env python3

import sys
import torch
import logging
import os
import platform
import psutil
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Validator for project environment and dependencies."""
    
    def __init__(self):
        self.required_python_version = (3, 8)
        self.required_packages = {
            'torch': '2.0.0',
            'torchvision': '0.15.0',
            'transformers': '4.30.0',
            'pillow': '9.5.0',
            'numpy': '1.24.0',
            'pandas': '2.0.0'
        }
        self.required_dirs = [
            'data/DeepFashion/images',
            'data/DeepFashion/annotations',
            'src/data',
            'src/models',
            'src/training',
            'src/utils',
            'tests/data',
            'tests/models',
            'tests/integration',
            'configs',
            'notebooks',
            'checkpoints',
            'logs'
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        current_version = sys.version_info[:2]
        if current_version < self.required_python_version:
            logger.error(
                f"Python version {self.required_python_version[0]}.{self.required_python_version[1]} "
                f"or higher is required. Found: {current_version[0]}.{current_version[1]}"
            )
            return False
        logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}")
        return True
    
    def check_packages(self) -> bool:
        """Check if required packages are installed with correct versions."""
        all_passed = True
        for package, version in self.required_packages.items():
            try:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    if current_version < version:
                        logger.error(
                            f"Package {package} version {version} or higher required. "
                            f"Found: {current_version}"
                        )
                        all_passed = False
                    else:
                        logger.info(f"Package {package} version check passed: {current_version}")
                else:
                    logger.warning(f"Could not determine version for package {package}")
            except ImportError:
                logger.error(f"Required package {package} is not installed")
                all_passed = False
        return all_passed
    
    def check_gpu_support(self) -> bool:
        """Check GPU support for M1 Mac."""
        if torch.backends.mps.is_available():
            logger.info("MPS (M1 GPU) support is available")
            return True
        else:
            logger.warning("MPS (M1 GPU) support is not available. Training will use CPU only.")
            return False
    
    def check_directory_structure(self) -> bool:
        """Check if required directories exist."""
        all_exist = True
        for dir_path in self.required_dirs:
            path = Path(dir_path)
            if not path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                all_exist = False
            else:
                logger.info(f"Directory check passed: {dir_path}")
        return all_exist
    
    def check_system_resources(self) -> Tuple[bool, Dict[str, float]]:
        """Check available system resources."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        resources = {
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_available_gb': disk.free / (1024**3)
        }
        
        # Define minimum requirements
        min_memory_gb = 8
        min_disk_gb = 50
        
        passed = (
            resources['memory_total_gb'] >= min_memory_gb and
            resources['disk_available_gb'] >= min_disk_gb
        )
        
        if passed:
            logger.info("System resources check passed:")
        else:
            logger.error("System resources check failed:")
            
        logger.info(f"Memory: {resources['memory_available_gb']:.1f}GB available "
                   f"of {resources['memory_total_gb']:.1f}GB total")
        logger.info(f"Disk: {resources['disk_available_gb']:.1f}GB available "
                   f"of {resources['disk_total_gb']:.1f}GB total")
        
        return passed, resources
    
    def check_file_permissions(self) -> bool:
        """Check if we have necessary file permissions."""
        test_dir = Path('test_permissions')
        try:
            # Test directory creation
            test_dir.mkdir(exist_ok=True)
            
            # Test file creation
            test_file = test_dir / 'test.txt'
            test_file.write_text('test')
            
            # Test file reading
            _ = test_file.read_text()
            
            # Clean up
            test_file.unlink()
            test_dir.rmdir()
            
            logger.info("File permissions check passed")
            return True
            
        except (OSError, PermissionError) as e:
            logger.error(f"File permissions check failed: {str(e)}")
            return False
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        checks = [
            (self.check_python_version(), "Python version"),
            (self.check_packages(), "Package versions"),
            (self.check_gpu_support(), "GPU support"),
            (self.check_directory_structure(), "Directory structure"),
            (self.check_file_permissions(), "File permissions")
        ]
        
        # System resources check
        resources_passed, _ = self.check_system_resources()
        checks.append((resources_passed, "System resources"))
        
        # Print summary
        logger.info("\nValidation Summary:")
        all_passed = True
        for passed, name in checks:
            status = "✓" if passed else "✗"
            logger.info(f"{status} {name}")
            all_passed = all_passed and passed
            
        return all_passed

def main():
    """Main validation function."""
    validator = EnvironmentValidator()
    if validator.validate_all():
        logger.info("\nAll validation checks passed!")
        sys.exit(0)
    else:
        logger.error("\nSome validation checks failed. Please fix the issues and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main() 