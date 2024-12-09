from setuptools import setup, find_packages

setup(
    name="style_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "pytest>=7.3.0",
        "black>=23.3.0"
    ],
    python_requires=">=3.9",
) 