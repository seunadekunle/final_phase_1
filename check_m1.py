import torch
import platform
import subprocess
import logging
from typing import Dict, Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class M1Checker:
    """Check M1 Mac configuration and optimization settings."""
    
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
        self.is_arm = platform.machine() == "arm64"
        
    def check_m1_compatibility(self) -> bool:
        """Check if running on M1 Mac."""
        if not (self.is_mac and self.is_arm):
            logger.warning("Not running on M1 Mac")
            return False
        logger.info("Running on M1 Mac")
        return True
    
    def check_mps_support(self) -> Dict[str, bool]:
        """Check MPS (Metal Performance Shaders) support."""
        results = {
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
            'mps_device_available': torch.backends.mps.is_available()
        }
        
        if results['mps_available']:
            logger.info("MPS support is available")
        else:
            logger.warning("MPS support is not available")
            
        return results
    
    def check_torch_version(self) -> Optional[str]:
        """Check PyTorch version for M1 support."""
        if not hasattr(torch, '__version__'):
            logger.error("Could not determine PyTorch version")
            return None
            
        version = torch.__version__
        required_version = '2.0.0'
        
        if version < required_version:
            logger.warning(
                f"PyTorch version {required_version} or higher recommended for M1 support. "
                f"Found: {version}"
            )
        else:
            logger.info(f"PyTorch version check passed: {version}")
            
        return version
    
    def check_memory_config(self) -> Dict[str, int]:
        """Check memory configuration."""
        try:
            # Get system memory info
            mem_cmd = "sysctl hw.memsize"
            mem_output = subprocess.check_output(mem_cmd.split()).decode()
            total_memory = int(mem_output.split()[1]) / (1024**3)  # Convert to GB
            
            # Get swap info
            swap_cmd = "sysctl vm.swapusage"
            swap_output = subprocess.check_output(swap_cmd.split()).decode()
            swap_total = float(swap_output.split()[2].replace('M', '')) / 1024  # Convert to GB
            
            memory_config = {
                'total_memory_gb': int(total_memory),
                'swap_memory_gb': int(swap_total)
            }
            
            logger.info(f"Memory configuration: {memory_config}")
            return memory_config
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get memory configuration: {str(e)}")
            return {'total_memory_gb': 0, 'swap_memory_gb': 0}
    
    def check_optimization_settings(self) -> Dict[str, bool]:
        """Check optimization settings for M1."""
        settings = {
            'mps_enabled': torch.backends.mps.is_available(),
            'deterministic': torch.backends.mps.deterministic,
            'benchmark_enabled': torch.backends.mps.benchmark
        }
        
        # Log recommendations
        if not settings['benchmark_enabled']:
            logger.info(
                "Consider enabling benchmark mode for better performance: "
                "torch.backends.mps.benchmark = True"
            )
            
        return settings
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations for M1 Mac."""
        suggestions = []
        
        if not torch.backends.mps.is_available():
            suggestions.append(
                "Install PyTorch with MPS support: "
                "pip3 install --pre torch torchvision torchaudio --extra-index-url "
                "https://download.pytorch.org/whl/nightly/cpu"
            )
            
        memory_config = self.check_memory_config()
        if memory_config['total_memory_gb'] < 16:
            suggestions.append(
                "Consider reducing batch size or using gradient accumulation "
                "due to limited memory"
            )
            
        if not torch.backends.mps.benchmark:
            suggestions.append(
                "Enable benchmark mode for better performance: "
                "torch.backends.mps.benchmark = True"
            )
            
        return suggestions
    
    def run_all_checks(self) -> Dict[str, any]:
        """Run all M1-specific checks."""
        if not self.check_m1_compatibility():
            return {'is_m1_mac': False}
            
        results = {
            'is_m1_mac': True,
            'mps_support': self.check_mps_support(),
            'torch_version': self.check_torch_version(),
            'memory_config': self.check_memory_config(),
            'optimization_settings': self.check_optimization_settings()
        }
        
        # Add optimization suggestions
        results['suggestions'] = self.suggest_optimizations()
        
        return results

def main():
    """Run M1 configuration checks."""
    checker = M1Checker()
    results = checker.run_all_checks()
    
    logger.info("\nM1 Configuration Summary:")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
        
    if results.get('suggestions'):
        logger.info("\nOptimization Suggestions:")
        for suggestion in results['suggestions']:
            logger.info(f"- {suggestion}")

if __name__ == '__main__':
    main() 