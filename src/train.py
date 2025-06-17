import os
import random
import numpy as np
import tensorflow as tf

from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_loader import load_data
from src.build_model import build_model

from .config import (
    SEED,
    BASELINE_MODEL_OUTPUT_PATH,
    STD_MODEL_OUTPUT_PATH,
    UNITS,
    EPOCHS,
    BATCH_SIZE
)


def train_model():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_test, y_train, y_test = load_data()

    mlp_base = KerasClassifier(
        model=build_model,
        model__input_dim=X_train.shape[1],
        model__units=UNITS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        random_state=SEED
    )

    mlp_base.fit(X_train, y_train)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', KerasClassifier(
            model=build_model,
            model__input_dim=X_train.shape[1],
            model__units=UNITS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            random_state=SEED
        ))
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs(BASELINE_MODEL_OUTPUT_PATH, exist_ok=True)
    os.makedirs(STD_MODEL_OUTPUT_PATH, exist_ok=True)

    joblib.dump(mlp_base, os.path.join(
        BASELINE_MODEL_OUTPUT_PATH, 'mlp_base.joblib'))
    joblib.dump(pipeline, os.path.join(
        STD_MODEL_OUTPUT_PATH, 'mlp_std.joblib'))

    print("\n✅ Training completed.")


if __name__ == "__main__":
    train_model()
