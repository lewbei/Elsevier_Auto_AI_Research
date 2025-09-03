# Elsevier Auto AI Research

An end-to-end automated AI research pipeline that finds relevant papers, identifies novel research directions, plans experiments, and conducts machine learning research with minimal human intervention.

## 🚀 Overview

This system automates the complete research workflow from literature review to experimental validation:

1. **Paper Discovery**: Automatically finds and downloads relevant research papers from Elsevier
2. **Literature Analysis**: Summarizes papers using Large Language Models (LLMs)
3. **Novelty Detection**: Identifies novel research directions and opportunities
4. **Experiment Planning**: Creates detailed experimental plans and specifications
5. **Automated Experimentation**: Runs machine learning experiments with multiple configurations
6. **Research Paper Generation**: Automatically drafts research papers with results

## ✨ Key Features

- **🔍 Intelligent Paper Discovery**: Leverages Elsevier API with relevance filtering
- **🤖 LLM-Powered Analysis**: Uses multiple LLM providers (OpenAI, DeepSeek) for paper summarization and analysis
- **🎯 Domain-Agnostic**: Configurable for any research domain through simple goal specification
- **🔬 Automated Experimentation**: Built-in PyTorch experiment runner with hyperparameter optimization
- **📊 Comprehensive Reporting**: Generates detailed reports, visualizations, and LaTeX papers
- **🖥️ Cross-Platform**: Windows + Git-Bash friendly, no shell dependencies
- **⚙️ Highly Configurable**: YAML-first configuration with environment variable overrides
- **🧪 Programmatic API**: No CLI required - pure Python entrypoints

## 📋 Prerequisites

- Python 3.8+
- Required API keys:
  - `ELSEVIER_KEY`: For paper discovery
  - `DEEPSEEK_API_KEY` or OpenAI API key: For LLM operations
  - `X_ELS_INSTTOKEN` (optional): For enhanced Elsevier access

> **⚠️ Important**: If you only have an `ELSEVIER_KEY` (without `X_ELS_INSTTOKEN`), you must access the API from within your institution's IP address range. The institution token allows access from any IP address.

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/lewbei/Elsevier_Auto_AI_Research.git
cd Elsevier_Auto_AI_Research
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Configure research settings** (optional):
```bash
# Edit config.yaml to customize your research domain and parameters
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your API credentials:

```env
ELSEVIER_KEY=your_elsevier_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
X_ELS_INSTTOKEN=your_elsevier_token  # optional - enables access from any IP
```

> **Note**: Without `X_ELS_INSTTOKEN`, you must run this from your institution's network to access Elsevier APIs.

### YAML Configuration

The main configuration is in `config.yaml`:

```yaml
project:
  goal: "your research objective"
  title: "optional explicit title"

dataset:
  name: your_dataset
  path: data/dataset
  kind: imagefolder  # imagefolder | cifar10 | custom

pipeline:
  skip:
    find_papers: false
    summaries: false
    novelty: false
    planner: false
    iterate: false
  max_iters: 2
  write_paper: true

llm:
  default: gpt-5-mini
  custom: deepseek
  use: default
```

### Dataset Support

The system supports multiple dataset formats:

- **ImageFolder**: Standard PyTorch ImageFolder format with train/val/test splits
- **CIFAR10**: Built-in torchvision CIFAR10 dataset
- **Custom**: User-defined dataset classes

## 🚀 Usage

### Basic Usage

Run the complete research pipeline:

```python
from agents import orchestrator
orchestrator.main()
```

Or use the convenience script:

```bash
python run_pipeline.py
```

### Programmatic Usage

Run individual pipeline stages:

```python
# Summarize papers
from agents.summarize import process_pdfs
process_pdfs("pdfs/", "data/summaries/")

# Generate novelty analysis
from agents.novelty import main as novelty_main
novelty_main()

# Create experimental plan
from agents.planner import main as planner_main
planner_main()

# Run experiments
from agents.iterate import iterate
iterate(novelty_dict, max_iters=2)
```

### Example: Skin Cancer Classification

```yaml
project:
  goal: "skin cancer classification and detection using deep learning"

dataset:
  name: isic
  path: data/isic
  kind: imagefolder
  splits:
    train: train
    val: val
    test: test

pipeline:
  max_iters: 3
  codegen:
    enable: true
```

## 📁 Project Structure

```
├── agents/                 # Main pipeline stages
│   ├── paper_finder.py    # Paper discovery and download
│   ├── summarize.py       # LLM-based paper summarization
│   ├── novelty.py         # Novelty detection and analysis
│   ├── planner.py         # Experiment planning
│   ├── iterate.py         # Automated experimentation
│   ├── write_paper.py     # Research paper generation
│   └── orchestrator.py    # Pipeline orchestration
├── lab/                   # Shared utilities
│   ├── config.py          # Configuration management
│   ├── experiment_runner.py # PyTorch experiment runner
│   ├── codegen_utils.py   # Code generation utilities
│   ├── mutations.py       # Hyperparameter mutations
│   └── logging_utils.py   # Logging and reporting
├── utils/                 # Core utilities
│   └── llm_utils.py       # LLM interface and caching
├── data/                  # Pipeline artifacts
├── pdfs/                  # Downloaded papers
├── runs/                  # Experiment results
├── paper/                 # Generated papers
├── tests/                 # Unit tests
├── config.yaml           # Main configuration
└── requirements.txt       # Python dependencies
```

## 📊 Outputs

The pipeline generates several types of outputs:

- **📄 Paper Summaries**: `data/summaries/*.json`
- **🎯 Novelty Analysis**: `data/novelty_report.json`
- **📋 Experiment Plan**: `data/plan.json`
- **📈 Experiment Results**: `runs/summary.json`, `runs/best.json`
- **📊 Visualizations**: `runs/accuracy.png`, `runs/dashboard.html`
- **📝 Research Paper**: `paper/paper.md`, `paper/main.tex`
- **💰 Cost Tracking**: `runs/llm_cost.json`

## 🔧 Advanced Features

### Multi-Agent Analysis

Enable multi-persona analysis for richer insights:

```yaml
pipeline:
  novelty:
    personas:
      enable: true
      steps: 4
      debate:
        enable: true
        rounds: 2
        roles: [PhD, Professor, Postdoc]
```

### Code Generation

Automatically generate custom PyTorch modules:

```yaml
pipeline:
  codegen:
    enable: true
    editor:
      enable: true  # Generate training hooks
```

### Human-in-the-Loop

Add confirmation gates for critical decisions:

```yaml
pipeline:
  hitl:
    confirm: true
    auto_approve: false
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Current test coverage includes:
- Configuration management
- Mutation generation
- Stage management
- Prompt overrides
- Paper generation
- Security validation

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)**: Detailed agent documentation
- **[docs/STRUCTURE.md](docs/STRUCTURE.md)**: Project structure overview
- **[docs/analysis_comparative_plan.md](docs/analysis_comparative_plan.md)**: Comparative analysis
- **[CODE_REVIEW_SUMMARY.md](CODE_REVIEW_SUMMARY.md)**: Recent code review findings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Guidelines

- Follow existing code style and patterns
- Add unit tests for new functionality
- Update documentation as needed
- Ensure cross-platform compatibility
- Use type hints where appropriate

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of PyTorch and torchvision
- Integrates with Elsevier ScienceDirect API
- Supports multiple LLM providers (OpenAI, DeepSeek)
- Inspired by the AI-Scientist and Agent Laboratory projects

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the [existing issues](https://github.com/lewbei/Elsevier_Auto_AI_Research/issues)
2. Create a new issue with detailed information
3. Include relevant configuration and error messages

### Common Issues

**Elsevier API Access Errors**: If you're getting authentication errors with the Elsevier API:
- Ensure you're using your institution's network if you don't have an `X_ELS_INSTTOKEN`
- Contact your institution's library to obtain an institution token for off-campus access
- Verify your API key is active and has not expired

## 🔮 Roadmap

- [ ] Support for additional dataset formats
- [ ] Integration with more LLM providers
- [ ] Enhanced visualization and reporting
- [ ] Multi-modal research capabilities
- [ ] Improved experiment management
- [ ] Web interface for pipeline monitoring

---

**Happy Researching! 🚀🔬**