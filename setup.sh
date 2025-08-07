#!/bin/bash

# setup.sh - Environment Setup Script for Employee Churn Prediction Agent
# This script sets up the complete development environment for the agent to run

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
PROJECT_NAME="employee-churn-sentiment"
VENV_NAME="employee-churn-env"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Banner
echo "=================================================="
echo "Employee Churn Prediction Agent Environment Setup"
echo "=================================================="
echo ""

# Check OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

log_info "Detected OS: ${MACHINE}"

# 1. Check and install system dependencies
log_info "Checking system dependencies..."

if [[ "$MACHINE" == "Mac" ]]; then
    # macOS setup
    if ! check_command "brew"; then
        log_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install system dependencies
    log_info "Installing system dependencies via Homebrew..."
    brew update
    brew install python@${PYTHON_VERSION} git curl wget jq tree
    
    # Install optional but useful tools
    brew install --cask docker
    
elif [[ "$MACHINE" == "Linux" ]]; then
    # Linux setup
    if check_command "apt"; then
        # Debian/Ubuntu
        log_info "Installing system dependencies via apt..."
        sudo apt update
        sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
                           python3-pip git curl wget jq tree build-essential \
                           libssl-dev libffi-dev libbz2-dev libreadline-dev \
                           libsqlite3-dev libncurses5-dev libncursesw5-dev \
                           xz-utils tk-dev libxml2-dev libxmlsec1-dev
    elif check_command "yum"; then
        # RHEL/CentOS/Fedora
        log_info "Installing system dependencies via yum/dnf..."
        sudo yum install -y python${PYTHON_VERSION} python3-pip git curl wget jq tree \
                           gcc gcc-c++ make openssl-devel bzip2-devel \
                           libffi-devel readline-devel sqlite-devel \
                           ncurses-devel xz-devel tk-devel
    fi
    
    # Install Docker (optional)
    if ! check_command "docker"; then
        log_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        log_warning "Please log out and back in for Docker permissions to take effect"
    fi
fi

# 2. Verify Python installation
log_info "Verifying Python installation..."
if check_command "python${PYTHON_VERSION}"; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif check_command "python3"; then
    PYTHON_CMD="python3"
    PYTHON_ACTUAL_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_ACTUAL_VERSION" != "$PYTHON_VERSION" ]]; then
        log_warning "Python version mismatch. Expected ${PYTHON_VERSION}, got ${PYTHON_ACTUAL_VERSION}"
    fi
elif check_command "python"; then
    PYTHON_CMD="python"
else
    log_error "Python not found. Please install Python ${PYTHON_VERSION}"
    exit 1
fi

log_success "Python found: $($PYTHON_CMD --version)"

# 3. Install/upgrade pip and essential Python packages
log_info "Upgrading pip and installing essential packages..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# 4. Install Poetry (recommended for dependency management)
log_info "Installing Poetry..."
if ! check_command "poetry"; then
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
    # Add Poetry to PATH
    if [[ "$MACHINE" == "Mac" ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    log_info "Poetry already installed. Updating..."
    poetry self update
fi

# 5. Create project directory structure
log_info "Creating project directory structure..."
if [[ ! -d "$PROJECT_NAME" ]]; then
    mkdir -p "$PROJECT_NAME"
fi

cd "$PROJECT_NAME"

# Create the full directory structure
mkdir -p employee_churn/{data,features,models,nlp,utils}
mkdir -p tests/{test_data,test_features,test_models,test_nlp,integration}
mkdir -p scripts
mkdir -p data/{raw,processed,external}
mkdir -p models/artifacts
mkdir -p configs
mkdir -p docs/{api,tutorials}
mkdir -p notebooks/exploratory

# Create __init__.py files
find employee_churn tests -type d -exec touch {}/__init__.py \;

log_success "Project directory structure created"

# 6. Initialize Poetry project
log_info "Initializing Poetry project..."
if [[ ! -f "pyproject.toml" ]]; then
    poetry init --name employee-churn-sentiment \
                --description "Employee churn prediction using structured data and sentiment analysis" \
                --author "Agent <agent@example.com>" \
                --python "^${PYTHON_VERSION}" \
                --no-interaction
fi

# 7. Add core dependencies
log_info "Adding core dependencies..."
poetry add pandas numpy scikit-learn xgboost lightgbm
poetry add transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry add spacy nltk textblob vadersentiment
poetry add pydantic click loguru python-dotenv
poetry add mlflow wandb optuna
poetry add matplotlib seaborn plotly
poetry add fastapi uvicorn
poetry add pyyaml toml

# Add development dependencies
poetry add --group dev pytest pytest-cov pytest-mock pytest-xdist
poetry add --group dev black isort flake8 mypy
poetry add --group dev pre-commit
poetry add --group dev sphinx sphinx-rtd-theme
poetry add --group dev jupyter ipykernel
poetry add --group dev bandit safety

log_success "Dependencies added to pyproject.toml"

# 8. Create and activate virtual environment
log_info "Creating virtual environment..."
poetry env use $PYTHON_CMD
poetry install

log_success "Virtual environment created and dependencies installed"

# 9. Install additional NLP models
log_info "Installing additional NLP models..."
poetry run python -m spacy download en_core_web_sm
poetry run python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# 10. Set up pre-commit hooks
log_info "Setting up pre-commit hooks..."
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

poetry run pre-commit install

# 11. Create configuration files
log_info "Creating configuration files..."

# .env.example
cat > .env.example << 'EOF'
# Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data Configuration
DATA_PATH=./data
MODEL_PATH=./models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# MLFlow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=employee_churn

# Database Configuration (if needed)
# DATABASE_URL=sqlite:///employee_churn.db

# External API Keys (if needed)
# OPENAI_API_KEY=your_key_here
# WANDB_API_KEY=your_key_here
EOF

# configs/model_config.yaml
cat > configs/model_config.yaml << 'EOF'
# Model Configuration
models:
  baseline:
    type: "random_forest"
    parameters:
      n_estimators: 100
      max_depth: 10
      random_state: 42
  
  enhanced:
    type: "xgboost"
    parameters:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
      random_state: 42

# Feature Engineering
features:
  structured:
    - tenure_months
    - department
    - salary_band
    - performance_score
    - promotions_count
  
  text_derived:
    - sentiment_score
    - emotion_scores
    - topic_indicators

# Training Configuration
training:
  test_size: 0.2
  cv_folds: 5
  scoring: "roc_auc"
EOF

# configs/logging_config.yaml
cat > configs/logging_config.yaml << 'EOF'
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.FileHandler
    filename: logs/app.log
    level: DEBUG
    formatter: default
root:
  level: INFO
  handlers: [console, file]
EOF

# 12. Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
data/raw/*
data/processed/*
models/artifacts/*
logs/
*.pkl
*.joblib
.DS_Store
mlflow.db
wandb/
EOF

# 13. Create basic README.md
cat > README.md << 'EOF'
# Employee Churn Prediction with Sentiment Analysis

A comprehensive Python package for predicting employee churn using both structured HR data and unstructured feedback analysis.

## Quick Start

1. Ensure environment is set up (run `./setup.sh` if not already done)
2. Activate the virtual environment: `poetry shell`
3. Install the package: `poetry install`
4. Run tests: `poetry run pytest`

## Development

This project uses Poetry for dependency management and pre-commit for code quality.

### Running Tests
```bash
poetry run pytest --cov=employee_churn
```

### Code Formatting
```bash
poetry run black .
poetry run isort .
poetry run flake8 .
```

## Project Structure

See `AGENTS.md` for detailed project structure and development guidelines.
EOF

# 14. Create logs directory
mkdir -p logs

# 15. Initialize git repository
log_info "Initializing git repository..."
if [[ ! -d ".git" ]]; then
    git init
    git add .
    git commit -m "Initial project setup"
fi

# 16. Create Docker configuration
log_info "Creating Docker configuration..."
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "employee_churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - ENVIRONMENT=development
    depends_on:
      - mlflow

  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow && 
               mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0"
    volumes:
      - ./mlflow:/mlflow
EOF

# 17. Final checks and summary
log_info "Running final checks..."

# Check if Poetry environment is working
if poetry run python --version >/dev/null 2>&1; then
    log_success "Poetry environment is working correctly"
else
    log_error "Poetry environment setup failed"
    exit 1
fi

# Check if core packages are installed
if poetry run python -c "import pandas, sklearn, transformers" >/dev/null 2>&1; then
    log_success "Core packages are installed and importable"
else
    log_error "Some core packages are not properly installed"
    exit 1
fi

echo ""
echo "=================================================="
log_success "Environment setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: poetry shell"
echo "2. Copy .env.example to .env and configure as needed"
echo "3. Review the project structure in AGENTS.md"
echo "4. Start development following the tasks outlined in AGENTS.md"
echo ""
echo "Useful commands:"
echo "- Run tests: poetry run pytest"
echo "- Format code: poetry run black ."
echo "- Type checking: poetry run mypy employee_churn/"
echo "- Start development server: poetry run uvicorn employee_churn.api:app --reload"
echo ""
echo "Happy coding! 🚀"
