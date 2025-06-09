# MLP Sonar Classifier

A simple project to train and evaluate two multilayer perceptron models on the Sonar data using TensorFlow, SciKeras, and Scikit-Learn — one without data standardization and another with standardized input data.

---

# Installation

1. Clone the repo

```bash
git clone https://github.com/yourusername/mlp-iris-classifier.git
cd mlp-sonar
```

2. Create a Conda enviornment

It is included an `environment.yml` for Conda users: 

```bash 
conda env create -f environment.yml
conda activate mlp-sonar
```

# Usage

1. Verify the dataset

The [Sonar Dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) from the UCI Machine Learning Repository is already included under data/sonar.csv.

2. Adjust settings

Open `config.yaml`and tweak any values you like (seed, test_size, units, etc.)

3. Run the full pipeline

```bash
python run_all.py
```

This will: 

- Train de MLP without data standardization and with standardized input data
- Save the two mlp models to `models` folder
- Evaluate and print train/test accuracy and sample predictions

# Project Structure

```
mlp-iris-classifier/
│
├── config.yaml          # Experiment settings
├── environment.yml      # Conda environment spec
│
├── data/
│   └── sonar.csv         # Sonar Dataset
│
├── models/              # (Auto-created) Trained model & params
│
├── src/
│   ├── config.py        # Loads config.yaml
│   ├── data_loader.py   # Reads & splits data
│   ├── model_builder.py # Defines the Keras MLP
│   ├── train.py         # Hyperparameter search & model saving
│   └── evaluate.py      # Loads model & prints metrics
│
└── run_all.py           # Runs train.py then evaluate.py
```

# Dependencies 

- Python 3.7+
- numpy, scikt-learn, tensorflow, scikeras, joblib, PyYAML

With Conda:

```bash 
conda env create -f environment.yml
conda activate mlp-sonar
```
