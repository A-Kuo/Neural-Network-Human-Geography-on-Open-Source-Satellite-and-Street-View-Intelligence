# INTEGRATION & PROFESSIONALIZATION PLAN
## Converting PhD Project → Production Package

**Purpose:** Transform this research project into a professional Python package that can be integrated with other repositories and deployed to production.

---

## PART I: CURRENT STATE ASSESSMENT

### What We Have (Research Phase)
- ✅ Isolated branch: `claude/neural-expressivity-geographic-zkPXj`
- ✅ Scripts in flat `data_pipeline/` directory
- ✅ Notebooks for experimentation
- ✅ Configuration in single `config.yaml`
- ✅ Tests focused on individual modules
- ✅ Fixed external dependencies (no API abstraction)

### What We Need (Production Phase)
- ❌ Importable package: `gnn_geography` (or similar)
- ❌ Standardized APIs for each component
- ❌ Dependency injection for external services
- ❌ Environment-aware configuration
- ❌ Integration tests across module boundaries
- ❌ Version pinning + lock file
- ❌ CI/CD pipeline for automated testing
- ❌ API documentation (Sphinx, auto-generated)
- ❌ Error handling with custom exceptions
- ❌ Logging configuration for production
- ❌ Docker/container support
- ❌ PyPI publishing capability

---

## PART II: PACKAGE STRUCTURE REFACTORING

### Current Structure (Research)
```
.
├── data_pipeline/
│   ├── fetch_*.py
│   ├── clean_*.py
│   ├── compute_*.py
│   ├── extract_*.py
│   ├── build_dataset.py
│   └── __init__.py (empty)
├── phase2_approximation/
│   ├── gnn_theory.py
│   ├── synthetic_tasks.py
│   └── *.ipynb
├── tests/
│   ├── test_data_pipeline.py
│   └── conftest.py
└── config.yaml
```

### Target Structure (Production)
```
gnn-geography/
├── README.md
├── pyproject.toml                    # NEW: PEP 517 project config
├── setup.py                          # NEW: Legacy support
├── setup.cfg                         # NEW: Configuration
├── requirements/                     # NEW: Organized dependencies
│   ├── base.txt                      # Core deps
│   ├── dev.txt                       # Testing, linting, docs
│   ├── prod.txt                      # Production extras
│   └── lock.txt                      # Pinned versions
├── src/gnn_geography/                # NEW: Source root (PEP 420)
│   ├── __init__.py
│   ├── __version__.py                # NEW: Single source of truth for version
│   ├── config.py                     # NEW: Configuration management
│   ├── logging_config.py             # NEW: Logging setup
│   ├── exceptions.py                 # NEW: Custom exceptions
│   ├── pipeline/                     # Refactored from data_pipeline/
│   │   ├── __init__.py
│   │   ├── fetchers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # NEW: Abstract base class
│   │   │   ├── census.py             # FROM: fetch_census.py
│   │   │   ├── osm.py                # FROM: fetch_osm.py
│   │   │   ├── transit.py            # FROM: fetch_gtfs.py
│   │   │   └── streetview.py         # FROM: fetch_streetview.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── osm.py                # FROM: clean_osm.py
│   │   │   ├── transit.py            # FROM: compute_transit.py
│   │   │   └── features.py           # FROM: extract_features.py
│   │   ├── builder.py                # FROM: build_dataset.py
│   │   └── pipeline.py               # NEW: Orchestrator
│   ├── models/                       # NEW: Model definitions
│   │   ├── __init__.py
│   │   ├── gnn.py                    # FROM: GNN classes
│   │   ├── training.py               # NEW: Training loop
│   │   └── evaluation.py             # NEW: Evaluation metrics
│   ├── theory/                       # FROM: phase2_approximation/
│   │   ├── __init__.py
│   │   ├── expressivity.py           # FROM: gnn_theory.py
│   │   └── synthetic.py              # FROM: synthetic_tasks.py
│   ├── utils/                        # NEW: Utilities
│   │   ├── __init__.py
│   │   ├── geometry.py               # Spatial operations
│   │   ├── graph.py                  # Graph construction
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── io.py                     # File I/O helpers
│   └── cli/                          # NEW: CLI interface
│       ├── __init__.py
│       └── main.py                   # Entry points
├── tests/
│   ├── conftest.py
│   ├── unit/                         # NEW: Organize by type
│   │   ├── test_pipeline.py
│   │   ├── test_models.py
│   │   ├── test_theory.py
│   │   └── test_utils.py
│   ├── integration/                  # NEW: End-to-end tests
│   │   └── test_pipeline_e2e.py
│   └── fixtures/                     # NEW: Shared test data
│       └── sample_data.py
├── docs/                             # NEW: Documentation
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   │   ├── pipeline.rst
│   │   ├── models.rst
│   │   └── theory.rst
│   ├── guides/
│   │   ├── quickstart.md
│   │   ├── configuration.md
│   │   └── integration.md
│   └── examples/                     # NEW: Jupyter notebooks as docs
│       ├── basic_usage.ipynb
│       └── chicago_experiment.ipynb
├── .github/
│   └── workflows/                    # NEW: CI/CD
│       ├── test.yml
│       ├── lint.yml
│       └── publish.yml
├── docker/                           # NEW: Containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── config/                           # NEW: Configuration files
│   ├── default.yaml
│   ├── chicago.yaml
│   └── test.yaml
├── Dockerfile                        # Keep for convenience
├── pyproject.toml                    # NEW: Modern Python packaging
├── tox.ini                           # NEW: Test environments
├── .pre-commit-config.yaml           # NEW: Pre-commit hooks
└── LICENSE                           # NEW: Explicit license
```

---

## PART III: PACKAGE ARCHITECTURE

### 1. **Core Package Module** (`src/gnn_geography/__init__.py`)

```python
"""
GNN Geography: Graph Neural Networks for Urban Analysis

Public API for external integrations.
"""

from . import pipeline, models, theory, utils
from .__version__ import __version__
from .config import Config
from .exceptions import GNNGeographyError

__all__ = [
    'pipeline',
    'models',
    'theory',
    'utils',
    'Config',
    'GNNGeographyError',
    '__version__',
]
```

### 2. **Configuration Management** (`src/gnn_geography/config.py`)

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml

@dataclass
class RegionConfig:
    name: str
    bbox: dict  # {north, south, east, west}
    utm_crs: str

@dataclass
class PipelineConfig:
    census_api_key: Optional[str] = None
    streetview_api_key: Optional[str] = None
    cache_dir: Path = Path('./cache')
    output_dir: Path = Path('./output')
    
class Config:
    """Environment-aware configuration."""
    
    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        with open(path) as f:
            return cls(**yaml.safe_load(f))
    
    @classmethod
    def from_env(cls, env: str = 'dev') -> 'Config':
        """Load config based on environment."""
        config_path = Path(__file__).parent.parent / 'config' / f'{env}.yaml'
        return cls.from_file(config_path)
```

### 3. **Custom Exceptions** (`src/gnn_geography/exceptions.py`)

```python
class GNNGeographyError(Exception):
    """Base exception for all GNN Geography errors."""
    pass

class DataSourceError(GNNGeographyError):
    """Error fetching/processing data source."""
    pass

class ValidationError(GNNGeographyError):
    """Data validation failed."""
    pass

class ConfigurationError(GNNGeographyError):
    """Configuration error."""
    pass

class APIError(GNNGeographyError):
    """External API error."""
    pass

class PrivacyViolationError(GNNGeographyError):
    """Privacy rule violation detected."""
    pass
```

### 4. **Logging Configuration** (`src/gnn_geography/logging_config.py`)

```python
import logging
from pathlib import Path

def setup_logging(log_dir: Path, level: str = 'INFO'):
    """Configure structured logging."""
    log_dir.mkdir(exist_ok=True)
    
    config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_dir / 'gnn_geography.log',
                'formatter': 'standard',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': level,
        },
    }
    logging.config.dictConfig(config)
```

---

## PART IV: API DESIGN

### 1. **Pipeline Orchestrator** (NEW)

```python
# gnn_geography/pipeline/pipeline.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class PipelineOutput:
    dataset: pd.DataFrame
    metadata: dict
    audit_logs: dict

class DataPipeline:
    """Orchestrator for end-to-end data processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetchers = {
            'census': CensusFetcher(config),
            'osm': OSMFetcher(config),
            'transit': TransitFetcher(config),
            'streetview': StreetViewFetcher(config),
        }
        self.processors = {
            'osm': OSMProcessor(),
            'transit': TransitProcessor(),
            'features': FeatureExtractor(),
        }
    
    def fetch_all(self) -> dict:
        """Fetch all data sources."""
        results = {}
        for name, fetcher in self.fetchers.items():
            results[name] = fetcher.fetch()
        return results
    
    def process_all(self, raw_data: dict) -> dict:
        """Process all data layers."""
        processed = {}
        processed['osm'] = self.processors['osm'].process(raw_data['osm'])
        processed['transit'] = self.processors['transit'].process(raw_data['transit'])
        processed['features'] = self.processors['features'].process(raw_data['streetview'])
        return processed
    
    def build(self, raw_data: dict) -> PipelineOutput:
        """Full pipeline: fetch → process → validate → output."""
        processed = self.process_all(raw_data)
        dataset = self._join_datasets(processed, raw_data['census'])
        
        # Privacy validation
        self._validate_privacy(dataset)
        
        return PipelineOutput(
            dataset=dataset,
            metadata={...},
            audit_logs={...},
        )
    
    def _validate_privacy(self, df: pd.DataFrame) -> None:
        """Check privacy rules."""
        if any(col in df.columns for col in ['latitude', 'longitude', 'lat', 'lon']):
            if 'median_household_income' in df.columns:
                raise PrivacyViolationError("lat/lon + income cannot coexist")
```

### 2. **Model Training API** (NEW)

```python
# gnn_geography/models/training.py

from typing import Tuple
import torch
from torch.utils.data import DataLoader

class GNNTrainer:
    """Standard training interface."""
    
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), **config.optimizer)
        self.criterion = torch.nn.MSELoss()
    
    def train(self, 
              train_data: DataLoader,
              val_data: DataLoader,
              epochs: int = 200) -> dict:
        """Train with early stopping."""
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training step
            train_loss = self._train_epoch(train_data)
            history['train_loss'].append(train_loss)
            
            # Validation step
            val_loss = self._validate_epoch(val_data)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        return history
    
    def evaluate(self, test_data: DataLoader) -> dict:
        """Evaluate on test set with metrics."""
        predictions = []
        actuals = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_data:
                pred = self.model(batch)
                predictions.append(pred)
                actuals.append(batch.y)
        
        predictions = torch.cat(predictions)
        actuals = torch.cat(actuals)
        
        return {
            'r2': r2_score(actuals, predictions),
            'rmse': torch.sqrt(self.criterion(predictions, actuals)).item(),
            'mae': torch.abs(predictions - actuals).mean().item(),
        }
```

### 3. **Unified Data Fetcher Interface** (NEW)

```python
# gnn_geography/pipeline/fetchers/base.py

from abc import ABC, abstractmethod
from typing import Any
import logging

class DataFetcher(ABC):
    """Base class for all data sources."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fetch(self) -> Any:
        """Fetch data from source."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate fetched data."""
        pass
    
    def fetch_with_retry(self, max_retries: int = 3) -> Any:
        """Fetch with automatic retry."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching (attempt {attempt + 1}/{max_retries})")
                data = self.fetch()
                if self.validate(data):
                    return data
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        raise APIError(f"Failed to fetch {self.__class__.__name__} after {max_retries} attempts")
```

---

## PART V: DEPENDENCY MANAGEMENT

### 1. **pyproject.toml** (NEW - PEP 517)

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gnn-geography"
version = "0.1.0"
description = "Graph Neural Networks for Urban Analysis"
authors = [{name = "A. Kuo", email = "akuo@..."}]
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24.0,<2.0.0",
    "pandas>=1.5.0,<2.1.0",
    "geopandas>=0.12.0,<0.15.0",
    "torch>=2.0.0,<3.0.0",
    "torch-geometric>=2.2.0,<2.5.0",
    "torchvision>=0.15.0,<0.18.0",
    "networkx>=3.0,<4.0",
    "scikit-learn>=1.2.0,<1.4.0",
    "requests>=2.28.0,<3.0.0",
    "loguru>=0.7.0,<0.8.0",
    "pydantic>=2.0.0,<3.0.0",
    "pyyaml>=6.0,<7.0",
    "tenacity>=8.2.0,<9.0.0",
    "shapely>=2.0,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0,<8.0",
    "pytest-cov>=4.0,<5.0",
    "pytest-xdist>=3.0,<4.0",
    "black>=23.0,<24.0",
    "isort>=5.12,<6.0",
    "ruff>=0.1.0,<0.2.0",
    "mypy>=1.0,<2.0",
    "pre-commit>=3.0,<4.0",
]
docs = [
    "sphinx>=6.0,<7.0",
    "sphinx-rtd-theme>=1.2,<2.0",
    "sphinx-autodoc-typehints>=1.20,<2.0",
]
prod = [
    "gunicorn>=20.1.0,<22.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]

[project.urls]
Repository = "https://github.com/a-kuo/gnn-geography"
Documentation = "https://gnn-geography.readthedocs.io"
Issues = "https://github.com/a-kuo/gnn-geography/issues"

[project.scripts]
gnn-geography = "gnn_geography.cli:main"

[tool.setuptools]
packages = ["gnn_geography"]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
```

### 2. **requirements/base.txt** (Locked versions)

```
numpy==1.24.3
pandas==1.5.3
geopandas==0.13.2
torch==2.0.1
torch-geometric==2.3.1
torchvision==0.15.2
scikit-learn==1.3.2
requests==2.31.0
loguru==0.7.2
pydantic==2.4.2
pyyaml==6.0.1
tenacity==8.2.3
shapely==2.0.1
```

---

## PART VI: TESTING & CI/CD

### 1. **Test Organization**

```
tests/
├── conftest.py                          # Shared fixtures
├── unit/
│   ├── test_config.py                   # Configuration
│   ├── test_exceptions.py               # Exception handling
│   ├── pipeline/
│   │   ├── test_fetchers.py             # Individual fetchers
│   │   ├── test_processors.py           # Individual processors
│   │   └── test_builder.py              # Dataset builder
│   ├── models/
│   │   ├── test_gnn.py                  # GNN models
│   │   ├── test_training.py             # Training loop
│   │   └── test_evaluation.py           # Evaluation metrics
│   ├── theory/
│   │   ├── test_expressivity.py         # WL theory
│   │   └── test_synthetic.py            # Synthetic tasks
│   └── utils/
│       ├── test_geometry.py
│       ├── test_graph.py
│       └── test_metrics.py
├── integration/
│   ├── test_pipeline_e2e.py             # Full pipeline
│   ├── test_training_e2e.py             # Train → eval
│   └── test_api_integration.py          # API calls
├── fixtures/
│   ├── sample_data.py                   # Test data
│   └── mock_apis.py                     # Mock external services
└── performance/                         # NEW: Performance benchmarks
    ├── test_pipeline_speed.py
    └── test_model_speed.py
```

### 2. **GitHub Actions CI/CD** (NEW)

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Lint with ruff
        run: ruff check src/ tests/
      
      - name: Format check with black
        run: black --check src/ tests/
      
      - name: Type check with mypy
        run: mypy src/
      
      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=src/gnn_geography \
            --cov-report=xml \
            --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## PART VII: INTEGRATION POINTS

### 1. **External Repository Integration**

```python
# How other repos would use this package

# Option A: Install as dependency
# In their pyproject.toml:
dependencies = [
    "gnn-geography @ git+https://github.com/a-kuo/gnn-geography@main"
]

# Option B: Direct import
from gnn_geography.pipeline import DataPipeline
from gnn_geography.models import GNNRegressor
from gnn_geography.config import Config

# Initialize
config = Config.from_file('config/chicago.yaml')
pipeline = DataPipeline(config)

# Fetch data
raw_data = pipeline.fetch_all()

# Process
result = pipeline.build(raw_data)

# Train model
from gnn_geography.models import GNNTrainer
trainer = GNNTrainer(model, config)
trainer.train(train_loader, val_loader)
metrics = trainer.evaluate(test_loader)
```

### 2. **REST API Server** (Optional - NEW)

```python
# gnn_geography/api/server.py

from fastapi import FastAPI, HTTPException
from gnn_geography.pipeline import DataPipeline
from gnn_geography.config import Config

app = FastAPI(title="GNN Geography API")

@app.post("/pipeline/build")
async def build_dataset(config_path: str) -> dict:
    """Trigger full pipeline."""
    try:
        config = Config.from_file(config_path)
        pipeline = DataPipeline(config)
        result = pipeline.build(pipeline.fetch_all())
        return {"status": "success", "n_rows": len(result.dataset)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/model/train")
async def train_model(config_path: str) -> dict:
    """Train GNN model."""
    # Implementation
    pass

@app.get("/model/predict")
async def predict(tract_id: str) -> dict:
    """Make prediction for tract."""
    # Implementation
    pass

# Run: uvicorn gnn_geography.api.server:app --reload
```

### 3. **CLI Interface** (NEW)

```python
# gnn_geography/cli/main.py

import click
from pathlib import Path
from gnn_geography.pipeline import DataPipeline
from gnn_geography.config import Config

@click.group()
def cli():
    """GNN Geography CLI."""
    pass

@cli.command()
@click.option('--config', type=click.Path(), default='config/chicago.yaml')
@click.option('--output', type=click.Path(), default='output/')
def build(config: str, output: str):
    """Build dataset."""
    cfg = Config.from_file(Path(config))
    pipeline = DataPipeline(cfg)
    result = pipeline.build(pipeline.fetch_all())
    result.dataset.to_parquet(f'{output}/dataset.parquet')
    click.echo(f"✓ Built dataset: {len(result.dataset)} rows")

@cli.command()
@click.option('--config', type=click.Path(), default='config/chicago.yaml')
@click.option('--depth', type=int, default=6)
@click.option('--epochs', type=int, default=200)
def train(config: str, depth: int, epochs: int):
    """Train GNN model."""
    # Implementation
    pass

if __name__ == '__main__':
    cli()

# Usage:
# gnn-geography build --config config/chicago.yaml
# gnn-geography train --depth 6 --epochs 200
```

---

## PART VIII: DOCUMENTATION & VERSIONING

### 1. **Sphinx Documentation** (NEW)

```
docs/
├── conf.py
├── index.rst
├── _static/
│   └── custom.css
├── _templates/
│   └── custom_layout.html
├── api/
│   ├── pipeline.rst
│   ├── models.rst
│   ├── theory.rst
│   └── utils.rst
├── guides/
│   ├── quickstart.md
│   ├── configuration.md
│   ├── integration.md
│   ├── troubleshooting.md
│   └── faq.md
└── examples/
    └── chicago_experiment.ipynb
```

### 2. **Version Management** (NEW)

```python
# src/gnn_geography/__version__.py

__version__ = "0.1.0"
__title__ = "gnn-geography"
__description__ = "Graph Neural Networks for Urban Analysis"
__author__ = "A. Kuo"
__license__ = "MIT"

# Single source of truth for version
# Used by: __init__.py, pyproject.toml, docs/conf.py
```

### 3. **Changelog** (NEW)

```markdown
# CHANGELOG

## [0.1.0] - 2026-04-15

### Added
- Initial release with Chicago dataset pipeline
- GNN models with variable depth (1-6 layers)
- Weisfeiler-Lehman expressivity theory
- Privacy-enforcing data builder
- Comprehensive test suite (52/52 passing)
- Documentation and API reference

### Changed
- Migrated from research scripts to package structure

### Fixed
- All 52 unit tests passing
- Code formatted with black/isort
- Type hints on all public APIs
```

---

## PART IX: DEPLOYMENT & CONTAINERIZATION

### 1. **Docker Build** (NEW)

```dockerfile
# docker/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

# Package
COPY . .
RUN pip install -e .

# Environment
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# CLI entry point
ENTRYPOINT ["gnn-geography"]
CMD ["--help"]
```

### 2. **Docker Compose** (NEW)

```yaml
# docker-compose.yml

version: '3.8'

services:
  pipeline:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./output:/app/output
    environment:
      CENSUS_API_KEY: ${CENSUS_API_KEY}
      STREETVIEW_API_KEY: ${STREETVIEW_API_KEY}
    command: build --config config/chicago.yaml

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      LOG_LEVEL: INFO
    command: uvicorn gnn_geography.api.server:app --host 0.0.0.0 --port 8000
```

---

## PART X: SECURITY & ENVIRONMENT MANAGEMENT

### 1. **Environment Variables** (NEW)

```bash
# .env.example
CENSUS_API_KEY=your_key_here
STREETVIEW_API_KEY=your_key_here
LOG_LEVEL=INFO
ENVIRONMENT=development
DATA_DIR=./data
OUTPUT_DIR=./output
```

### 2. **Secrets Management** (NEW)

```python
# gnn_geography/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    census_api_key: str = ""
    streetview_api_key: str = ""
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage: settings = Settings()
# Loads from .env, env vars, then defaults
```

---

## PART XI: WORK BREAKDOWN

### Phase 1: Package Restructuring (3-4 weeks)
- [ ] Create `src/gnn_geography/` directory structure
- [ ] Refactor `data_pipeline/` modules → `src/gnn_geography/pipeline/`
- [ ] Create base classes (DataFetcher, DataProcessor)
- [ ] Implement configuration management
- [ ] Implement custom exceptions
- [ ] Add logging configuration
- [ ] Create `src/gnn_geography/__init__.py` with public API
- [ ] Move notebooks → `docs/examples/`

### Phase 2: API Design & Orchestration (2-3 weeks)
- [ ] Create DataPipeline orchestrator
- [ ] Create GNNTrainer interface
- [ ] Implement unified fetcher/processor interfaces
- [ ] Add dependency injection for external services
- [ ] Write integration tests
- [ ] Document public APIs with docstrings

### Phase 3: Testing & Quality (2 weeks)
- [ ] Reorganize tests (unit/ vs. integration/)
- [ ] Create test fixtures and mock APIs
- [ ] Add performance benchmarks
- [ ] Achieve 90%+ code coverage
- [ ] Set up pre-commit hooks
- [ ] Add type checking (mypy strict mode)

### Phase 4: Documentation & Publishing (2 weeks)
- [ ] Set up Sphinx documentation
- [ ] Write API reference docs
- [ ] Write integration guides
- [ ] Create example notebooks
- [ ] Write README with quickstart
- [ ] Set up PyPI publishing

### Phase 5: Deployment & CI/CD (2 weeks)
- [ ] Create GitHub Actions workflows
- [ ] Set up Docker builds
- [ ] Create docker-compose for local dev
- [ ] Implement REST API (optional)
- [ ] Set up automated code quality checks
- [ ] Configure branch protection rules

### Phase 6: Integration Readiness (1 week)
- [ ] Write integration documentation
- [ ] Create example repos showing integration
- [ ] Set up shared fixtures for other repos
- [ ] Document version pinning strategy
- [ ] Create contribution guidelines

**Total Effort:** ~12-14 weeks

---

## PART XII: INTEGRATION WITH OTHER REPOS

### Scenario 1: Data Processing Team Uses Pipeline

```python
# In external repo: github.com/myorg/data-platform

from gnn_geography.pipeline import DataPipeline
from gnn_geography.config import Config

config = Config.from_file('config/chicago.yaml')
pipeline = DataPipeline(config)

# Fetch all data
raw = pipeline.fetch_all()

# Process
result = pipeline.build(raw)

# Output to their data warehouse
result.dataset.to_sql('neighborhood_features', engine)
```

### Scenario 2: ML Team Trains Models

```python
# In external repo: github.com/myorg/ml-pipeline

from gnn_geography.models import GNNRegressor, GNNTrainer
from gnn_geography.theory import compute_wl_dimension

# Load data from previous stage
data = pd.read_parquet('data/chicago_dataset.parquet')

# Check expressivity requirements
wl_dim, _ = compute_wl_dimension(graph)
print(f"Minimum GNN depth required: {wl_dim}")

# Train model
model = GNNRegressor(in_features=2056, hidden_dim=64, depth=6)
trainer = GNNTrainer(model, config)
metrics = trainer.train(train_loader, val_loader)
```

### Scenario 3: Research Team Validates Theory

```python
# In external repo: github.com/myorg/research-validation

from gnn_geography.theory import (
    weisfeiler_lehman_iteration,
    compute_wl_dimension,
    synthetic_income_function,
)

# Validate WL expressivity on new graph
graph = load_graph('new_city.graphml')
wl_dim, history = compute_wl_dimension(graph)

# Test on synthetic task
task = synthetic_income_function(graph, num_hops=5)
results = evaluate_gnn_on_synthetic_task(model, graph, task)
```

---

## PART XIII: SUCCESS METRICS

### Code Quality
- ✅ 90%+ code coverage
- ✅ Zero type errors (mypy strict)
- ✅ Zero style violations (black, ruff)
- ✅ All tests passing on Python 3.10, 3.11, 3.12

### Documentation
- ✅ API docs auto-generated from docstrings
- ✅ Integration guide with examples
- ✅ Quickstart under 5 minutes
- ✅ All public functions documented

### Usability
- ✅ Installable via pip
- ✅ Can be imported as module
- ✅ CLI interface available
- ✅ Configuration via YAML/env vars

### Integration
- ✅ Used by at least 2 other internal repos
- ✅ Version pinning strategy documented
- ✅ Breaking changes tracked in changelog
- ✅ Deprecation warnings for API changes

---

## NEXT IMMEDIATE STEPS

1. **Week 1:** Create package structure (Phase 1, weeks 1)
2. **Week 2:** Refactor data_pipeline modules
3. **Week 3:** Implement orchestrators and APIs
4. **Week 4:** Reorganize tests and add integration tests
5. **Week 5:** Set up documentation and CI/CD

---

**This transforms the research project into an enterprise-ready package suitable for integration with other systems.**
