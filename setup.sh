#!/bin/bash

# Exit on error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check Python version
check_python_version() {
    required_version="3.8"
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        error "Python version $required_version or higher is required. Found: $python_version"
    fi
    log "Python version check passed: $python_version"
}

# Check system dependencies
check_dependencies() {
    log "Checking system dependencies..."
    
    # Check for required commands
    commands=("python3" "pip3" "git")
    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is required but not installed."
        fi
    done
    
    # Check for GPU support
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null; then
        log "MPS (M1 GPU) support detected"
    else
        warn "MPS (M1 GPU) support not detected. Training will use CPU only."
    fi
}

# Create project structure
create_directory_structure() {
    log "Creating project directory structure..."
    
    directories=(
        "data/DeepFashion/images"
        "data/DeepFashion/annotations"
        "src/data"
        "src/models"
        "src/training"
        "src/utils"
        "tests/data"
        "tests/models"
        "tests/integration"
        "configs"
        "notebooks"
        "checkpoints"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir" || error "Failed to create directory: $dir"
    done
    
    # Create necessary files
    touch logs/training.log
    touch logs/testing.log
    
    log "Directory structure created successfully"
}

# Set up virtual environment
setup_virtual_environment() {
    log "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        warn "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv || error "Failed to create virtual environment"
    source venv/bin/activate || error "Failed to activate virtual environment"
    
    log "Upgrading pip..."
    pip install --upgrade pip || error "Failed to upgrade pip"
    
    log "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt || error "Failed to install dependencies"
    else
        error "requirements.txt not found"
    fi
}

# Validate installation
validate_installation() {
    log "Validating installation..."
    
    # Activate virtual environment
    source venv/bin/activate || error "Failed to activate virtual environment"
    
    # Check if key packages are installed
    packages=("torch" "torchvision" "transformers" "pytest")
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            error "Package '$package' is not properly installed"
        fi
    done
    
    # Run basic tests
    if [ -f "pytest.ini" ]; then
        log "Running basic tests..."
        pytest tests/data/test_parser.py -v || warn "Some tests failed"
    else
        warn "pytest.ini not found, skipping tests"
    fi
}

# Main setup process
main() {
    log "Starting Fashion Style Classifier setup..."
    
    # Check requirements
    check_python_version
    check_dependencies
    
    # Create structure
    create_directory_structure
    
    # Set up environment
    setup_virtual_environment
    
    # Validate setup
    validate_installation
    
    log "Setup completed successfully!"
    log "To activate the virtual environment, run: source venv/bin/activate"
}

# Run main setup
main 