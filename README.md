# Plant Disease Detection

## What This Project Does

I built a machine learning model that predicts whether plants will get sick based on environmental conditions like temperature, humidity, rainfall, and soil pH., from a Kaggle Dataset. The model gets it right about 86% of the time, which is pretty decent for this kind of problem. I was looking for a dataset that was interesting and my sister has a hydrophonics farm which measures water ph, so I thought it was a good start. I do apologize that it is a Kaggle dataset. It's a big deal in lettuce hydrophonics about humidity and temps since it can kill or let them dry out.

## Data

You can grab the dataset from my Google Drive: https://drive.google.com/drive/folders/1igO5WPGRts_gz9jL-rjeptZD-Vs82dRS?usp=sharing, it is also available in Kaggle: https://www.kaggle.com/datasets/turakut/plant-disease-classification/data

The data has:
- **temperature, humidity, rainfall, soil_pH** (the environmental factors)
- **disease_present** (0 = healthy, 1 = diseased)
- About 76% healthy plants, 24% diseased

## How to Set Everything Up

### What You Need
- Python 3.9 or newer
- Git
- Docker Desktop (for Airflow orchestration)

### Getting Started
```bash
# Clone this repo
git clone https://github.com/JCarlo-Ona/59e69b0a433cbfce6ed2c895d787d30b558a3448b477b3787415ad2f3023d21e_plant_disease_detect.git
cd 59e69b0a433cbfce6ed2c895d787d30b558a3448b477b3787415ad2f3023d21e_plant_disease_detect

# Install UV if you don't have it
pip install uv

# Set up the environment (this installs all the packages)
uv sync

# Test that everything worked
uv run python -c "import pandas, sklearn; print('All good!')"

# Set up code quality checks
uv run pre-commit install
```

### Add the Dataset
Download the CSV from the Google Drive link above and put it in `data/raw/`

### Run Everything
```bash
uv run python src/run_pipeline.py
```

This will process the data, train the model, and test it.

## 🚀 ML Pipeline Orchestration with Airflow

### Pipeline Overview
The project uses Apache Airflow to orchestrate a complete ML workflow:

```
preprocess_data → train_model → evaluate_model → pipeline_summary
```

### Task Details
1. **preprocess_data**: Loads and splits the plant disease dataset
2. **train_model**: Trains multiple models and selects the best performer
3. **evaluate_model**: Evaluates the trained model on test data
4. **pipeline_summary**: Displays comprehensive results summary

### Running the Pipeline

#### Start Airflow (when Docker Desktop works)
```bash
# Initialize Airflow
docker-compose up airflow-init

# Start all services
docker-compose up -d

# Access Airflow UI
# http://localhost:8080
# Password's in the files

#### Manual Pipeline Execution
```bash
# Run the complete pipeline
python src/run_pipeline.py
```

### Pipeline Features
- ✅ **Task Dependencies**: Ensures proper execution order
- ✅ **XCom Communication**: Shares data between tasks
- ✅ **Error Handling**: Robust error management
- ✅ **Logging**: Comprehensive task logging
- ✅ **Manual Triggers**: On-demand pipeline execution

### Monitoring
- View task logs in Airflow UI
- Monitor task duration and success rates
- Track pipeline performance over time

## How I Organized Everything

```
├── README.md                   # This file
├── pyproject.toml              # Lists all the packages needed
├── .pre-commit-config.yaml     # Code formatting rules
├── docker-compose.yml          # Airflow orchestration
├── .env                        # Environment variables
├── data/
│   ├── raw/                    # Original dataset goes here
│   └── processed/              # Cleaned data gets saved here
├── src/
│   ├── data_preprocessing.py   # Loads and cleans the data
│   ├── feature_engineering.py # Creates new features from the original ones
│   ├── model_training.py       # Trains different ML models
│   ├── evaluation.py           # Tests how good the model is
│   └── run_pipeline.py         # Runs everything in order
├── models/
│   └── plant_disease_model.pkl # The trained model
├── deploy/                     # Deployment configs
│   ├── airflow/
│   │   ├── dags/              # Airflow DAGs
│   │   └── logs/              # Pipeline logs
│   └── docker/                # Docker configs
├── config/                     # Configuration files
└── reports/
    └── metrics.txt             # Performance results
```

## Code Quality Setup

I set up some automatic code checking with these rules:

- **trailing-whitespace & end-of-file-fixer:** Cleans up formatting to avoid annoying Git conflicts
- **check-yaml:** Makes sure config files don't have syntax errors
- **check-added-large-files:** Stops me from accidentally uploading huge files to GitHub
- **ruff:** Catches Python coding mistakes and keeps the style consistent
- **black:** Automatically formats Python code so it all looks the same

## Running the Code

You can either run everything at once:
```bash
uv run python src/run_pipeline.py
```

Or run each step individually if you want to see what's happening:
```bash
uv run python src/data_preprocessing.py
uv run python src/model_training.py
uv run python src/evaluation.py
```

## How Well It Works

The Random Forest model performed best:
- **86% accuracy**
- **78% precision**
- **59% recall**

There is a summary of it in `reports/metrics.txt`.

## Something that's was hard.

The trickiest part was getting UV to work properly due to: "TypeError: Author #1 must be an inline table" when trying to run `uv sync`. I think to add anything with Whitespace or solving the pre-commit hooks was hard.

Airflow and DAG was also a pain to fix. Apparently my BIOS had its CPU Virtualization off so Docker Desktop and Airflow wouldn't work. It's also downloading the dependencies not on the image but in the containers so it was longer to run.

We had to troubleshoot `PYTHONPATH` and Docker volumes carefully to make sure code ran as expected.
