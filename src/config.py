import os
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), os.pardir, 'config.yaml')


def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_config = load_config()

DATA_PATH = _config.get('data_path', 'data/sonar.csv')
BASELINE_MODEL_OUTPUT_PATH = _config.get('model_output_path', 'models')
STD_MODEL_OUTPUT_PATH = _config.get('model_output_path', 'models')

SEED = _config.get('seed', 42)
TRAIN_SIZE = _config.get('train_size', 0.3)

UNITS = _config.get("model", {}).get("units", 60)
EPOCHS = _config.get("train", {}).get("epochs", 100)
BATCH_SIZE = _config.get("train", {}).get("batch_size", 5)
