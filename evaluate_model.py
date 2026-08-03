import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
def load_model():

    model = joblib.load(
        "model_artifacts/best_model.pkl"
    )

    print("Best model loaded successfully.")

    return model

def load_dataset():

    csv_files = [
        f for f in os.listdir("synthetic_data")
        if f.endswith(".csv")
    ]

    dataset = pd.read_csv(
        os.path.join("synthetic_data", csv_files[0])
    )

    return dataset

def prepare_data(df):

    columns_to_drop = [
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "attack_type"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )
    from sklearn.preprocessing import LabelEncoder

    X = df.drop("label", axis=1)

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["label"])

    return X, y

def evaluate_model(model, X, y):
    """
    Evaluate the trained model.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y,
        predictions,
        average="macro"
    )
    recall = recall_score(
        y,
        predictions,
        average="macro"
    )
    f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y, predictions))

    return predictions

def save_confusion_matrix(y, predictions):

    os.makedirs("evaluation_results", exist_ok=True)

    cm = confusion_matrix(y, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.savefig(
        "evaluation_results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved.")

def main():

    model = load_model()

    dataset = load_dataset()

    X, y = prepare_data(dataset)

    predictions = evaluate_model(
        model,
        X,
        y
    )

    save_confusion_matrix(
        y,
        predictions
    )


if __name__ == "__main__":
    main()