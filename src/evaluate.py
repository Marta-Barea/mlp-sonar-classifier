import os
import joblib

from .data_loader import load_data
from .config import BASELINE_MODEL_OUTPUT_PATH, STD_MODEL_OUTPUT_PATH


def evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    base_mlp = joblib.load(os.path.join(
        BASELINE_MODEL_OUTPUT_PATH, 'mlp_base.pkl'))
    std_mlp = joblib.load(os.path.join(STD_MODEL_OUTPUT_PATH, 'mlp_std.pkl'))

    if not base_mlp or not std_mlp:
        print("❌ Model not found. Make sure to run src/train.py first.")
        return

    train_acc_base = base_mlp.score(X_train, y_train)
    test_acc_base = base_mlp.score(X_test, y_test)

    print(f"\n📊 Base Model Train accuracy: {train_acc_base * 100:.2f}%")
    print(f"📊 Base Model Test accuracy:  {test_acc_base * 100:.2f}%")

    train_acc_std = std_mlp.score(X_train, y_train)
    test_acc_std = std_mlp.score(X_test, y_test)

    print(f"\n📊 Standardized Model Train accuracy: {train_acc_std * 100:.2f}%")
    print(f"📊 Standardized Model Test accuracy:  {test_acc_std * 100:.2f}%")

    y_pred_base = base_mlp.predict(X_test)
    y_pred_std = std_mlp.predict(X_test)
    print("\n🔎 First 10 predictions vs. actual values (Base Model):")
    for i in range(10):
        print(
            f"   • Predicted: {int(y_pred_base[i])}, Actual: {int(y_test[i])}")

    print("\n🔎 First 10 predictions vs. actual values (Standardized Model):")
    for i in range(10):
        print(
            f"   • Predicted: {int(y_pred_std[i])}, Actual: {int(y_test[i])}")


if __name__ == "__main__":
    evaluate_model()
