# AGENTS.md

## 👷 Agent Role: Package Builder for Employee Churn Prediction

You are a code generation agent tasked with creating a modular Python package for employee churn prediction using structured HR data and unstructured feedback with sentiment analysis. Your job is to produce production-grade code in a clean repository structure using best practices.

---

## 🎯 Objectives

- Build a complete, testable, and documented Python package
- Implement churn prediction using both structured and text-derived features
- Use sentiment analysis and emotional cues from feedback to improve prediction quality
- Ensure the package is reproducible, extendable, and aligned with standard MLOps practices
- Maintain data privacy and ethical AI considerations throughout the development process

---

## 📦 Package Name

`employee_churn`

---

## 🗂️ Package Structure

```text
employee-churn-sentiment/
├── employee_churn/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── clean.py
│   │   └── validate.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer_structured.py
│   │   ├── engineer_text.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── registry.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── sentiment.py
│   │   ├── emotion.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   ├── test_nlp/
│   └── integration/
├── scripts/
│   ├── train_model.py
│   ├── predict_risk.py
│   └── evaluate_model.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   └── artifacts/
├── configs/
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── logging_config.yaml
├── docs/
│   ├── api/
│   └── tutorials/
├── notebooks/
│   └── exploratory/
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── CHANGELOG.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── .pre-commit-config.yaml
```

---

## 🔨 Tasks

### Phase 1: Foundation & Setup
- [ ] **Project Initialization**
  - [ ] Create the main module `employee_churn/` with proper `__init__.py` files
  - [ ] Set up `pyproject.toml` with Poetry including development dependencies
  - [ ] Configure `.gitignore`, `README.md`, and `LICENSE`
  - [ ] Set up pre-commit hooks for code quality
  - [ ] Create Docker configuration for consistent environments

- [ ] **Configuration Management**
  - [ ] Implement `config.py` with environment-aware settings
  - [ ] Create YAML config files for different components
  - [ ] Add environment variable handling with `.env` support

### Phase 2: Data Pipeline
- [ ] **Data Loading & Validation** 
  - [ ] Implement `load.py` for HRIS and feedback data ingestion with multiple format support
  - [ ] Implement `clean.py` to sanitize and normalize both structured and unstructured data
  - [ ] Add `validate.py` for data quality checks and schema validation
  - [ ] Implement data versioning and lineage tracking

- [ ] **Feature Engineering**
  - [ ] `engineer_structured.py`: tenure, team changes, promotion history, performance metrics
  - [ ] `engineer_text.py`: sentiment polarity, emotion categories, topic modeling, text statistics
  - [ ] `feature_store.py`: centralized feature management and caching
  - [ ] Implement feature importance tracking and monitoring

### Phase 3: NLP & Text Analysis
- [ ] **Sentiment & Emotion Analysis**
  - [ ] `sentiment.py`: multi-model sentiment analysis (VADER, TextBlob, transformers)
  - [ ] `emotion.py`: emotion detection using state-of-the-art models
  - [ ] `preprocessing.py`: text cleaning, tokenization, and normalization
  - [ ] Add support for multiple languages if needed

### Phase 4: Model Development
- [ ] **Training & Evaluation**
  - [ ] `train.py`: baseline models + text-enhanced models with hyperparameter tuning
  - [ ] `evaluate.py`: comprehensive metrics, fairness checks, model explainability (SHAP, LIME)
  - [ ] `predict.py`: batch and individual churn risk scoring with confidence intervals
  - [ ] `registry.py`: model versioning and artifact management

### Phase 5: Automation & Deployment
- [ ] **Scripts & CLI**
  - [ ] `train_model.py`: CLI for model training with configurable parameters
  - [ ] `predict_risk.py`: CLI for batch and individual predictions
  - [ ] `evaluate_model.py`: CLI for model evaluation and comparison
  - [ ] Add argument validation and error handling

### Phase 6: Testing & Quality Assurance
- [ ] **Comprehensive Testing**
  - [ ] Unit tests for all modules (`tests/test_*.py`)
  - [ ] Integration tests for end-to-end workflows
  - [ ] Performance benchmarking tests
  - [ ] Data quality and model bias tests
  - [ ] Achieve >90% code coverage

### Phase 7: Documentation & Deployment
- [ ] **Documentation**
  - [ ] API documentation with Sphinx
  - [ ] Usage tutorials and examples
  - [ ] Model cards for transparency
  - [ ] Contributing guidelines

- [ ] **Deployment Preparation**
  - [ ] CI/CD pipeline with GitHub Actions
  - [ ] Model monitoring and alerting setup
  - [ ] Security scanning and dependency management

---

## 🛠️ Technical Requirements

### Core Dependencies
- **Python**: 3.10+
- **ML Stack**: scikit-learn, pandas, numpy, xgboost/lightgbm
- **NLP**: transformers, spacy, nltk, textblob
- **Infrastructure**: pydantic, click, loguru, mlflow
- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: black, isort, flake8, mypy

### Data Requirements
- **Structured Data**: Employee demographics, performance, tenure, compensation
- **Unstructured Data**: Exit interviews, survey feedback, performance reviews
- **Privacy**: Implement data anonymization and GDPR compliance features

### Performance Requirements
- **Training**: Support datasets up to 100K+ employees
- **Inference**: <500ms response time for individual predictions
- **Scalability**: Horizontal scaling for batch processing

---

## ⚠️ Constraints & Guidelines

### Code Quality Standards
- Use Python 3.10+ with type hints throughout
- Follow PEP 8 style guidelines with Black formatting
- No hardcoded file paths; use configuration management
- Implement comprehensive error handling and logging
- All functions must have docstrings (Google style)
- Code coverage must exceed 90%

### Security & Privacy
- Implement data encryption for sensitive information
- Add audit logging for model predictions
- Ensure GDPR compliance with data handling
- Use environment variables for sensitive configurations

### MLOps Best Practices
- Version control for data, code, and models
- Implement model drift detection
- Add bias detection and fairness metrics
- Provide model explainability features
- Include A/B testing framework for model comparison

---

## ✅ Completion Criteria

### Functional Requirements
- [ ] All modules implemented, tested, and documented
- [ ] Package installable with `poetry install`
- [ ] All CLI scripts functional with help documentation
- [ ] Model achieves baseline performance metrics (AUC > 0.75)
- [ ] Data pipeline handles various input formats reliably

### Quality Requirements
- [ ] >90% test coverage across all modules
- [ ] All type hints properly implemented
- [ ] Pre-commit hooks passing
- [ ] Documentation complete and accessible
- [ ] Docker containers build and run successfully

### Delivery Requirements
- [ ] README with clear installation and usage instructions
- [ ] CHANGELOG tracking all major changes
- [ ] Model cards documenting model behavior and limitations
- [ ] Performance benchmarks documented

---

## 🔄 Development Workflow

### Branch Strategy
```bash
# Feature development
git checkout -b feature/module-name
git add .
git commit -m "feat: add module-name with tests and documentation"
git push origin feature/module-name

# Create pull request
gh pr create --title "Add [module-name]" --body "Description of changes"
```

### Code Review Checklist
- [ ] All tests passing
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance impact assessed

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create release tag: `git tag v0.1.0`
4. Build and publish package
5. Update `ROADMAP.md` with completed milestones

---

## 📈 Success Metrics

### Technical Metrics
- **Model Performance**: AUC-ROC > 0.80, Precision > 0.75, Recall > 0.70
- **Code Quality**: Test coverage > 90%, Zero critical security issues
- **Performance**: Training time < 30 minutes, Inference < 500ms

### Business Metrics
- **Accuracy**: Correctly identify 80%+ of at-risk employees
- **Actionability**: Provide interpretable insights for HR interventions
- **Scalability**: Support enterprise-scale deployments (10K+ employees)

---

## 🚀 Future Enhancements (Post v0.1.0)

- Real-time prediction API with FastAPI
- Advanced NLP with custom transformer models
- Automated retraining pipelines
- Integration with popular HRIS systems
- Advanced visualization dashboard
- Multi-language support for global organizations

---

## 📞 Support & Maintenance

For questions or issues during development:
1. Check existing documentation and tests
2. Review similar implementations in the codebase
3. Create detailed GitHub issues with reproducible examples
4. Follow the established coding standards and patterns
