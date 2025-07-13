# Plant Disease Detection

## What This Project Does

I built a machine learning model that predicts whether plants will get sick based on environmental conditions like temperature, humidity, rainfall, and soil pH. The model gets it right about 86% of the time, which is pretty decent for this kind of problem.

I picked this dataset because plant diseases are a real problem for farmers, and being able to predict them early could help save crops. Plus, it makes sense that things like humidity and temperature would affect whether plants get diseases - anyone who's dealt with garden mold knows this!

## Getting the Data

You can grab the dataset from my Google Drive: https://drive.google.com/drive/folders/1igO5WPGRts_gz9jL-rjeptZD-Vs82dRS?usp=sharing

Just download the `plant_disease_dataset.csv` file and put it in the `data/raw/` folder. It's about 10,000 rows of plant data with environmental measurements.

If you want to load it programmatically:
```python
import pandas as pd
df = pd.read_csv("data/raw/plant_disease_dataset.csv")
```

The data has:
- **temperature, humidity, rainfall, soil_pH** (the environmental factors)
- **disease_present** (0 = healthy, 1 = diseased)
- About 76% healthy plants, 24% diseased

## How to Set Everything Up

### What You Need
- Python 3.9 or newer
- Git

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

## How I Organized Everything

```
├── README.md                    # This file
├── pyproject.toml              # Lists all the packages needed
├── .pre-commit-config.yaml     # Code formatting rules
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
└── reports/
    └── metrics.txt             # Performance results
```

I separated raw and processed data because you never want to mess with the original data - if something goes wrong, you can always start over. Each Python file does one specific thing, which makes it easier to debug and test individual parts.

The models folder is where the trained AI "brain" gets saved, and reports is for the performance numbers that show how well it works.

## Code Quality Setup

I set up some automatic code checking with these rules:

- **trailing-whitespace & end-of-file-fixer:** Cleans up formatting to avoid annoying Git conflicts
- **check-yaml:** Makes sure config files don't have syntax errors
- **check-added-large-files:** Stops me from accidentally uploading huge files to GitHub
- **ruff:** Catches Python coding mistakes and keeps the style consistent
- **black:** Automatically formats Python code so it all looks the same

Basically, these tools make the code look professional and catch mistakes before they become problems. Every time I commit code, they automatically check and fix formatting issues.

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
- **86% accuracy** - gets it right 86 out of 100 times
- **78% precision** - when it says a plant is diseased, it's right 78% of the time
- **59% recall** - catches about 59% of all diseased plants

You can see all the detailed numbers in `reports/metrics.txt`.

## Something That Went Wrong (And How I Fixed It)

The trickiest part was getting UV to work properly. I kept getting this weird error: "TypeError: Author #1 must be an inline table" when trying to run `uv sync`.

Turns out the `pyproject.toml` file is really picky about formatting. I had written:
```toml
authors = ["My Name <email@example.com>"]
```

But it wanted:
```toml
authors = [{name = "My Name", email = "email@example.com"}]
```

Small difference, but it completely broke the setup! After some googling and reading error messages carefully, I figured out that TOML (the config file format) has very specific rules about how to structure data. Now I know to read documentation more carefully when setting up new tools.

## What I'd Do Next

If I had more time, I'd try:
- Adding more features like seasonal patterns or plant types
- Testing other ML algorithms like XGBoost
- Building a simple web app where farmers could input their conditions and get predictions
- Testing the model on real farm data to see if it actually works in practice
