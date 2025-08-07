#!/bin/bash

# setup.sh - Minimal Environment Setup for Employee Churn Agent
# Creates only the development environment, letting the agent handle project structure

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Banner
echo -e "${GREEN}Employee Churn Agent Environment Setup${NC}"
echo "Setting up minimal development environment..."
echo

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)  MACHINE="Linux";;
    Darwin*) MACHINE="Mac";;
    *)       MACHINE="Unknown";;
esac
log_info "Detected OS: ${MACHINE}"

# 1. Install system dependencies
log_info "Installing system dependencies..."
if [[ "$MACHINE" == "Mac" ]]; then
    if ! check_command "brew"; then
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install python@${PYTHON_VERSION} git curl jq || true
    
elif [[ "$MACHINE" == "Linux" ]]; then
    if check_command "apt"; then
        sudo apt update
        sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
                           git curl jq build-essential || true
    elif check_command "yum"; then
        sudo yum install -y python${PYTHON_VERSION//./} python3-pip git curl jq \
                           gcc gcc-c++ make || true
    fi
fi

# 2. Verify Python
if check_command "python${PYTHON_VERSION}"; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif check_command "python3"; then
    PYTHON_CMD="python3"
else
    log_error "Python ${PYTHON_VERSION} not found!"
    exit 1
fi
log_success "Python: $($PYTHON_CMD --version)"

# 3. Install Poetry
if ! check_command "poetry"; then
    log_info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    export PATH="$HOME/.local/bin:$PATH"
    # Add to shell profile
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 2>/dev/null || true
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
fi

# 4. Upgrade pip
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# 5. Install spaCy model (commonly needed for NLP tasks)
log_info "Installing spaCy English model..."
$PYTHON_CMD -m pip install spacy
$PYTHON_CMD -m spacy download en_core_web_sm 2>/dev/null || log_warning "Could not download spaCy model (will install later)"

# 6. Create minimal .env if needed
if [[ ! -f ".env" ]]; then
    cat > .env << 'EOF'
# Development Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
    log_info "Created basic .env file"
fi

# 7. Create minimal .gitignore if needed
if [[ ! -f ".gitignore" ]]; then
    cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
.venv/
venv/
data/
*.db
*.log
.DS_Store
mlflow.db
wandb/
EOF
    log_info "Created basic .gitignore"
fi

# Final message
echo
log_success "Environment setup complete! 🚀"
echo
echo "Ready for agent development. The agent will:"
echo "• Create the full project structure"
echo "• Set up Poetry with all dependencies"  
echo "• Build the complete employee churn system"
echo
echo "All necessary tools are installed and ready."
