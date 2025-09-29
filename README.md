# LLM Chemistry Estimation for Multi-LLM Recommendation

## Prelims

### LLM Chemistry code

Git clone the project and create a conda environment:

```sh
# clone project
› git clone https://github.com/SRI-CSL/llm-chemistry.git
› cd llm-chemistry
```

### LLM Chemistry Virtual Environment

We recommend running LLM Chemistry in a virtual environment to isolate its dependencies. Please ensure miniconda or anaconda is installed and initialized before proceeding. Two installation options for miniconda on macOS are shown below.

#### Option 1: Install Miniconda on macOS (using Homebrew)

```shell
› brew install --cask miniconda
```

#### Option 2: Install Miniconda on macOS (manual)

> **Note:** It is recommended to install Miniconda outside the `collm` project directory to avoid potential conflicts and ensure a clean project environment.

```shell
# install miniconda (instructions from website)
# You can also use homebrew to install miniconda
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# activate conda
source ~/miniconda3/bin/activate

# initialize all available shells
conda init --all
```

After installing and initializing conda, create the CoLLM environment:

```shell
# create conda environment
› ./scripts/conda_create_env.sh
› conda activate chemistry
```

## Running Examples

### Predefined Examples

You can run LLM Chemistry on a few pre-defined examples, or on actual performance histories (`runs*.csv` files) using the provided scripts. For example, to run the examples:

```sh
# Homogeneously bad performing models
› python examples.py --verbose --choice "worst"
# or Homogeneously good performing models
› python examples.py --verbose --choice "best"
# or Random performing models
› python examples.py --verbose --choice "random"
```

## Run LLM Chemistry for Multi-LLM Recommendation

You can start by running `--help` for the multi-LLM recommendation:

```sh
› python main.py --help
usage: main.py [-h] [--verbose] {recommend,rank,eval} ...

Run LLM Chemistry Estimation for Multi-LLM Recommendation.

positional arguments:
  {recommend,rank,eval}
    recommend           recommend optimal model subsets for a given task.
    rank                list top-K models by performance from the CSV file.
    eval                evaluates LLM Chemistry-based recommendation.

options:
  -h, --help            show this help message and exit
  --verbose, -v         print detailed results and intermediate steps.
```

### Multi-LLM Recommendation

Then, you can run the `recommend` command to get model recommendations for a specific task.
There is some level of randomness in the recommendations, so you may get different results on different runs. For example:

```sh
# Classification task
› python main.py recommend --data "runs.csv"
Setting: runs.csv as trial data file.
Setting: 0.5 as usage threshold.
----------------------------------------
Recommended subset for 'classify a short statement into a category of fakeness':
  1. claude-3-5-sonnet-latest
  2. claude-3-7-sonnet-20250219
  3. firefunction-v2
  4. gpt-4o
  5. llama3.3:latest
  6. mixtral:8x22b
  7. o1-mini
  8. o3-mini
  9. o4-mini
  10. qwen2.5:32b
# Automated Program Repair
› python main.py recommend --data "runs2.csv"
Setting: runs2.csv as trial data file.
Setting: 0.5 as usage threshold.
----------------------------------------
Recommended subset for 'fix buggy program':
  1. claude-3-5-sonnet-latest
  2. claude-3-7-sonnet-20250219
  3. firefunction-v2
  4. gpt-4o
  5. llama3.1:70b
  6. mixtral:8x22b
  7. o1-mini
  8. o3-mini
  9. qwen2.5:32b
# Clinical Note Summarization
› python main.py recommend --data "runs3.csv"
Setting: runs3.csv as trial data file.
Setting: 0.5 as usage threshold.
----------------------------------------
Recommended subset for 'generate a concise, clinically formal summary':
  1. claude-3-5-sonnet-latest
  2. claude-3-7-sonnet-20250219
  3. firefunction-v2
  4. gemini-2.0-flash
  5. gpt-4o
  6. llama3.1:70b
  7. llama3.3:latest
  8. mixtral:8x22b
  9. qwen2.5:32b
```

## Running Experiments

You can run the experiments through the `main.py` script:

```sh
# Research Question 1:
› python main.py eval --question rq1
# Research Question 2:
› python main.py eval --question rq2
# Research Question 3:
› python main.py eval --question rq3
```

### 🤝 Contributions

We have a [set of guidelines](CONTRIBUTING.md) for contributing to `llm-chemistry`.  These are guidelines, not rules. Use your best judgment, and feel free
to propose  changes to this document in a pull request.

## ⚖️ License

This project is under the TBD License. See the [LICENSE](https://github.com/SRI-CSL/llm-chemistry/blob/main/LICENSE) file for the full license text.
