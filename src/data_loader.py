import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATA_PATH, TRAIN_SIZE, SEED


def load_data():
    data_sonar = pd.read_csv(DATA_PATH, header=None)
    arr = data_sonar.values

    X_sonar = arr[:, :-1].astype(float)
    y_sonar = arr[:, -1]

    label_encoder = LabelEncoder()
    y_sonar = label_encoder.fit_transform(y_sonar)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sonar, y_sonar, test_size=TRAIN_SIZE, random_state=SEED
    )

    return X_train, X_test, y_train, y_test
