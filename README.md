# Elsevier Auto AI Research

An end-to-end automated AI research pipeline that discovers, analyzes, and conducts machine learning research using LLMs and academic APIs. This Windows-friendly system automates the entire research process from paper discovery to experiment execution and paper writing.

## 🚀 Features

- **Automated Paper Discovery**: Uses Elsevier API to find and download relevant academic papers
- **LLM-Powered Analysis**: Leverages DeepSeek API for paper summarization, novelty analysis, and research planning
- **Automated Experimentation**: Runs machine learning experiments with multiple datasets (ISIC, CIFAR10)
- **Research Paper Generation**: Automatically generates research papers in Markdown and LaTeX formats
- **Windows-Friendly**: Designed to work seamlessly on Windows with Git Bash
- **Configurable Pipeline**: Extensive environment variables for customization
- **Offline Fallbacks**: Graceful degradation when APIs are unavailable
- **Caching Support**: Built-in LLM response caching to reduce API costs

## 📋 Prerequisites

- **Python 3.8+** (Windows compatible)
- **Git Bash** (recommended for Windows)
- **API Keys**:
  - Elsevier API key (`ELSEVIER_KEY`)
  - DeepSeek API key (`DEEPSEEK_API_KEY`)
- **Optional**: X-ELS-Insttoken for institutional access

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lewbei/Elsevier_Auto_AI_Research.git
   cd Elsevier_Auto_AI_Research
   ```

2. **Install dependencies**:
   ```bash
   # Windows (recommended)
   cmd.exe /C "python -m pip install -r requirements.txt --disable-pip-version-check"
   
   # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   ELSEVIER_KEY=your_elsevier_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   X_ELS_INSTTOKEN=your_institutional_token  # optional
   ```

4. **Verify installation**:
   ```bash
   # Compile sources to check for syntax errors
   cmd.exe /C "python -c \"import sys,compileall; sys.exit(0 if compileall.compile_dir('.', force=True, quiet=1) else 1)\""
   
   # Run tests
   cmd.exe /C "python -m pytest -q"
   ```

## 🏃‍♂️ Quick Start

### Run the Complete Pipeline

```bash
# Run all steps (paper writing only if WRITE_PAPER=1)
cmd.exe /C "python run_pipeline.py"
```

### Run Individual Steps

1. **Find and Download Papers**:
   ```bash
   cmd.exe /C "python paper_finder"
   ```
   Output: `abstract_screen_deepseek.csv` and `pdfs/*.pdf`

2. **Analyze Novelty**:
   ```bash
   cmd.exe /C "python agents_novelty.py"
   ```
   Output: `data/novelty_report.json`

3. **Generate Research Plan**:
   ```bash
   cmd.exe /C "python agents_planner.py"
   ```
   Output: `data/plan.json`

4. **Run Experiments**:
   ```bash
   cmd.exe /C "python agents_iterate.py"
   ```
   Output: `runs/`, `experiments/`, `runs/summary.json`, `runs/dashboard.html`

5. **Generate Paper** (optional):
   ```bash
   cmd.exe /C "set WRITE_PAPER=1&& python agents_write_paper.py"
   ```
   Output: `paper/paper.md`, `paper/main.tex`

## ⚙️ Configuration

The system is highly configurable through environment variables:

### Required Variables
- `ELSEVIER_KEY`: Your Elsevier API key
- `DEEPSEEK_API_KEY`: Your DeepSeek API key

### Optional Variables
- `X_ELS_INSTTOKEN`: Institutional token for Elsevier access
- `WRITE_PAPER=1`: Enable paper drafting step
- `DATASET=isic|cifar10`: Choose dataset (default: isic)
- `ALLOW_FALLBACK_DATASET=true`: Use synthetic data if real dataset missing
- `ALLOW_DATASET_DOWNLOAD=true`: Allow CIFAR10 download
- `TIME_BUDGET_SEC=0`: Time budget in seconds (0 = unlimited)
- `HITL_CONFIRM=0`: Disable human-in-the-loop confirmations
- `LLM_CACHE=true`: Enable LLM response caching
- `LLM_CACHE_DIR=.cache/llm`: Cache directory location

## 📁 Project Structure

```
├── agents_novelty.py       # Paper analysis and novelty detection
├── agents_planner.py       # Research plan generation
├── agents_iterate.py       # Experiment execution engine
├── agents_write_paper.py   # Paper writing automation
├── paper_finder            # Paper discovery and download
├── run_pipeline.py         # Complete pipeline orchestrator
├── llm_utils.py           # LLM interaction utilities
├── pdf_utils.py           # PDF processing utilities
├── lab/                   # Laboratory utilities and mutations
├── docs/                  # Analysis and planning documents
├── tests/                 # Unit tests
├── data/                  # Generated data and reports
├── runs/                  # Experiment results and dashboards
├── paper/                 # Generated research papers
└── pdfs/                  # Downloaded academic papers
```

## 🔬 How It Works

1. **Paper Discovery**: Searches Elsevier database for relevant papers using configurable queries
2. **Content Analysis**: Extracts and summarizes paper content using LLM
3. **Novelty Detection**: Identifies research gaps and potential innovations
4. **Research Planning**: Generates actionable research plans with hypotheses and success criteria
5. **Experiment Execution**: Runs automated ML experiments with baseline, novelty, and ablation studies
6. **Result Analysis**: Aggregates results and generates visualizations
7. **Paper Writing**: Automatically drafts research papers based on findings

## 🎯 Datasets Supported

- **ISIC**: Skin cancer classification dataset (default)
- **CIFAR10**: Standard computer vision benchmark
- **Synthetic Data**: Fallback option for testing without real datasets

## 📊 Output Artifacts

- **Papers**: Downloaded PDFs and generated research papers
- **Data**: Summaries, novelty reports, and research plans
- **Experiments**: Complete experiment logs and results
- **Visualizations**: Accuracy plots and interactive dashboards
- **Reports**: HTML dashboards and JSON summaries

## 🔧 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file contains required API keys
2. **Rate Limits**: Built-in backoff mechanisms handle API rate limits
3. **Dataset Issues**: Use `ALLOW_FALLBACK_DATASET=true` for synthetic data
4. **Windows Path Issues**: Use `cmd.exe /C python ...` commands as shown

### Safe Rollback Options

- **Offline Planning**: Automatic fallback when LLM APIs fail
- **Compute Budget**: Set `TIME_BUDGET_SEC` to limit execution time
- **Minimal Runs**: Use `REPEAT_N=1` and `MUTATE_K=0` for quick tests
- **Cache Clearing**: Remove `.cache/llm` to refresh LLM responses

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
cmd.exe /C "python -m pytest -q"
```

Tests cover core functionality without hitting external APIs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Documentation

- [AGENTS.md](AGENTS.md): Detailed setup and usage instructions
- [docs/](docs/): Analysis and comparison documents
- [Comparative Analysis](docs/analysis_comparative_plan.md): Comparison with other research automation tools

## 🔗 Related Work

This project builds upon and compares with:
- [AI-Scientist v2](https://github.com/SakanaAI/AI-Scientist): Automated scientific research
- [Agent Laboratory](https://github.com/SakanaAI/AI-Scientist): Multi-agent research workflows

## 📧 Support

For questions, issues, or contributions, please open an issue on GitHub.

---

**Note**: This tool is designed for academic research automation. Ensure compliance with your institution's research policies and API usage terms.