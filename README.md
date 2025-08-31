# Elsevier Auto AI Research

An automated research pipeline that discovers, analyzes, and conducts AI research using multi-agent systems and large language models.

## Overview

This project provides an end-to-end automated research workflow that:

- **Discovers relevant papers** from Elsevier's ScienceDirect database
- **Analyzes literature** for novelty and research gaps using LLM agents
- **Generates research plans** automatically based on identified opportunities
- **Conducts experiments** with configurable datasets and models
- **Writes research papers** in Markdown and LaTeX formats

The system is designed to be domain-agnostic and can be adapted for various research areas by configuring the project goal and dataset.

## Features

- **Multi-agent architecture** with specialized agents for different research phases
- **Configurable datasets** supporting ImageFolder, CIFAR10, and custom loaders
- **LLM integration** with DeepSeek API and response caching
- **Experiment management** with PyTorch, including mutation and ablation studies
- **Automated reporting** with HTML dashboards and accuracy plots
- **Paper generation** with LaTeX table formatting and bibliography management
- **Windows support** with Git Bash compatibility
- **Fallback modes** for offline operation and missing dependencies

## Prerequisites

### Required API Keys

- **Elsevier API Key**: Required for accessing ScienceDirect papers
- **DeepSeek API Key**: Required for LLM-based analysis and planning

### Environment

- Python 3.8+ (tested with Python 3.12)
- Git Bash (for Windows users)
- Internet connection (for paper discovery and LLM calls)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lewbei/Elsevier_Auto_AI_Research.git
   cd Elsevier_Auto_AI_Research
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # ELSEVIER_KEY=your_elsevier_api_key
   # DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

## Configuration

### Basic Configuration (config.yaml)

Edit `config.yaml` to customize your research project:

```yaml
project:
  goal: "your research objective"
  title: "Optional paper title"

dataset:
  kind: imagefolder  # or cifar10, custom
  name: default
  path: data/dataset
  allow_fallback: false
  allow_download: false
```

### Environment Variables

Key environment variables (set in `.env`):

- `ELSEVIER_KEY`: Your Elsevier API key (required)
- `DEEPSEEK_API_KEY`: Your DeepSeek API key (required)
- `X_ELS_INSTTOKEN`: Optional institutional token
- `WRITE_PAPER=1`: Enable paper drafting step
- `DATASET=isic|cifar10`: Override dataset selection
- `ALLOW_FALLBACK_DATASET=true`: Use synthetic data if real dataset missing
- `TIME_BUDGET_SEC=0`: Set time budget (0 = unlimited)

See `AGENTS.md` for complete configuration options.

## Usage

### Quick Start

Run the complete pipeline:

```bash
python run_pipeline.py
```

### Step-by-Step Execution

1. **Find and download papers**:
   ```bash
   python -m agents.paper_finder
   ```

2. **Analyze novelty**:
   ```bash
   python -m agents.novelty
   ```

3. **Generate research plan**:
   ```bash
   python -m agents.planner
   ```

4. **Run experiments**:
   ```bash
   python -m agents.iterate
   ```

5. **Write paper draft** (optional):
   ```bash
   WRITE_PAPER=1 python -m agents.write_paper
   ```

### Windows Users

Use `cmd.exe` for running Python commands:

```cmd
cmd.exe /C "python run_pipeline.py"
```

## Project Structure

```
├── agents/              # Multi-agent pipeline components
│   ├── paper_finder.py  # Paper discovery and relevance filtering
│   ├── novelty.py       # Literature analysis and novelty synthesis
│   ├── planner.py       # Research plan generation
│   ├── iterate.py       # Experiment execution and iteration
│   ├── write_paper.py   # Paper drafting and formatting
│   └── stage_manager.py # Staged execution management
├── lab/                 # Shared utilities and libraries
│   ├── config.py        # YAML-first configuration loader
│   ├── experiment_runner.py # PyTorch experiment framework
│   ├── codegen_utils.py # Safe code generation utilities
│   └── ...              # Additional utilities
├── utils/               # Core utilities
│   ├── llm_utils.py     # LLM interaction and caching
│   └── pdf_utils.py     # PDF processing utilities
├── data/                # Pipeline artifacts and JSON outputs
├── paper/               # Generated paper drafts (Markdown/LaTeX)
├── runs/                # Experiment results and reports
├── tests/               # Unit tests
├── docs/                # Additional documentation
├── config.yaml          # Project configuration
└── run_pipeline.py      # Main pipeline runner
```

## Output Files

After running the pipeline, you'll find:

- **PDFs**: `pdfs/` - Downloaded research papers
- **Summaries**: `data/summaries/` - Paper summaries and critiques
- **Novelty Report**: `data/novelty_report.json` - Identified themes and ideas
- **Research Plan**: `data/plan.json` - Generated research plan
- **Experiment Results**: `runs/` - Results, reports, and visualizations
- **Paper Draft**: `paper/` - Generated paper in Markdown and LaTeX (if enabled)

## Examples

### Basic Research Pipeline

```bash
# Set your research goal
echo 'project:
  goal: "deep learning for medical image classification"
dataset:
  kind: cifar10
  allow_download: true' > config.yaml

# Run the complete pipeline
python run_pipeline.py
```

### Custom Dataset

```yaml
# config.yaml
dataset:
  kind: imagefolder
  path: path/to/your/dataset
  splits:
    train: training
    val: validation
    test: testing
```

### Quick Experiment-Only Run

```bash
# Skip paper finding (use existing PDFs)
SKIP_FIND_PAPERS=1 python run_pipeline.py
```

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Compile all Python sources to check for syntax errors:

```bash
python -c "import sys,compileall; sys.exit(0 if compileall.compile_dir('.', force=True, quiet=1) else 1)"
```

## Documentation

- **AGENTS.md**: Detailed setup and configuration guide
- **docs/STRUCTURE.md**: Project architecture overview
- **docs/**: Additional analysis and comparison documents
- **CODE_REVIEW_SUMMARY.md**: Recent code quality improvements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `python -m pytest`
6. Submit a pull request

## Troubleshooting

### Common Issues

- **Missing API keys**: Set `ELSEVIER_KEY` and `DEEPSEEK_API_KEY` in `.env`
- **Dataset not found**: Set `ALLOW_FALLBACK_DATASET=true` for synthetic data
- **Windows path issues**: Use `cmd.exe /C python ...` commands
- **Rate limits**: Built-in backoff retry logic handles API rate limits

### Offline Mode

The system supports offline operation with fallback modes:

- Missing papers: Use existing PDFs in `pdfs/` directory
- No LLM access: Planner falls back to offline planning mode
- No dataset: Enable `ALLOW_FALLBACK_DATASET=true` for synthetic data

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project integrates with:

- **Elsevier ScienceDirect API** for academic paper access
- **DeepSeek API** for large language model capabilities
- **PyTorch** for deep learning experiments
- **PyYAML** for configuration management