import os
import warnings
import joblib
import yaml

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df

def validate_data(df):
    """
    Validate the dataset before training.
    """

    print("\n" + "=" * 50)
    print("Data Validation")
    print("=" * 50)

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check for duplicate records
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Records: {duplicates}")

    # Display data types
    print("\nData Types:")
    print(df.dtypes)

    # Basic dataset information
    print("\nDataset Information:")
    print(df.info())

    return df

def load_config(config_path="config.yaml"):
    """
    Load configuration settings from a YAML file.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("\nConfiguration loaded successfully.")

    return config

def get_dataset_path(dataset_folder):
    """
    Find the first CSV file inside the dataset folder.
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_folder}"
        )

    csv_files = [
        file for file in os.listdir(dataset_folder)
        if file.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No CSV dataset found inside the dataset folder."
        )

    csv_path = os.path.join(dataset_folder, csv_files[0])

    print(f"\nDataset selected: {csv_path}")

    return csv_path

def preprocess_data(df, config):
    """
    Preprocess the dataset for machine learning.
    """

    print("\n" + "=" * 50)
    print("Data Preprocessing")
    print("=" * 50)

    # Remove unnecessary columns
    columns_to_drop = [
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "attack_type"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Separate features and target
    X = df.drop("label", axis=1)
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["label"])

    # Identify feature types
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    print(f"\nCategorical Features: {categorical_features}")
    print(f"Numerical Features: {numerical_features}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # Split dataset
    # First split (85% train+validation, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,
                           test_size=config["split"]["test_size"],
                       random_state=config["random_state"],
                     stratify=y
)

# Second split (70% train, 15% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=config["random_state"],
        stratify=y_train_val
)

    print(f"\nTraining Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Testing Samples: {len(X_test)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
)

from sklearn.feature_selection import mutual_info_classif
def feature_selection(X_train, y_train, preprocessor):
    """
    Perform feature selection using Mutual Information.
    """

    print("\n" + "=" * 50)
    print("Feature Selection")
    print("=" * 50)

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)

    # Calculate feature importance
    scores = mutual_info_classif(
        X_train_processed,
        y_train
    )

    # Feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    feature_scores = pd.DataFrame({
        "Feature": feature_names,
        "Score": scores
    })

    feature_scores = feature_scores.sort_values(
        by="Score",
        ascending=False
    )

    print("\nTop 10 Features:")
    print(feature_scores.head(10))
    # Save feature importance
    os.makedirs("model_artifacts", exist_ok=True)

    feature_scores.to_csv(
    "model_artifacts/feature_importance.csv",
     index=False
)

    print("\nFeature importance saved to:")
    print("model_artifacts/feature_importance.csv")

    return preprocessor, feature_scores
def train_random_forest(
    X_train,
    y_train,
    preprocessor,
    config
):
    """
    Train Random Forest model.
    """

    print("\n" + "=" * 50)
    print("Training Random Forest")
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=config["random_state"]
            )
        )
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["random_state"]
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_
def train_xgboost(
    X_train,
    y_train,
    preprocessor,
    config
):
    """
    Train XGBoost model.
    """

    print("\n" + "=" * 50)
    print("Training XGBoost")
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                random_state=config["random_state"],
                eval_metric="logloss"
            )
        )
    ])

    param_grid = {
        "classifier__learning_rate": [0.1],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["random_state"]
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_

def train_neural_network(
    X_train,
    y_train,
    preprocessor,
    config
):
    """
    Train Neural Network model.
    """

    print("\n" + "=" * 50)
    print("Training Neural Network")
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            MLPClassifier(
                random_state=config["random_state"]
            )
        )
    ])

    param_grid = {
        "classifier__hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64)
        ],
        "classifier__activation": [
            "relu",
            "tanh"
        ],
        "classifier__max_iter": [
            300
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["random_state"]
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_

def compare_models(
    rf_model,
    xgb_model,
    X_test,
    y_test
):
    """
    Compare trained models and select the best one.
    """

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

    results = []

    best_model = None
    best_score = 0

    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average="macro"
        )
        recall = recall_score(
            y_test,
            predictions,
            average="macro"
        )
        f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        print(f"\n{name}")
        print(classification_report(y_test, predictions))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        if f1 > best_score:
            best_score = f1
            best_model = model

    results_df = pd.DataFrame(results)

    os.makedirs("model_artifacts", exist_ok=True)

    results_df.to_csv(
        "model_artifacts/training_results.csv",
        index=False
    )

    return best_model, results_df

def save_best_model(best_model, preprocessor):
    """
    Save the best model and preprocessing pipeline.
    """

    os.makedirs("model_artifacts", exist_ok=True)

    joblib.dump(
        best_model,
        "model_artifacts/best_model.pkl"
    )

    joblib.dump(
        preprocessor,
        "model_artifacts/preprocessor.pkl"
    )

    print("\nBest model saved successfully.")
    print("Files saved in model_artifacts/")
def main():

    config = load_config()

    dataset_path = get_dataset_path(
        config["dataset"]["folder"]
    )

    df = load_data(dataset_path)

    validate_data(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = preprocess_data(df, config)

    feature_selection(
        X_train,
        y_train,
        preprocessor
    )
    rf_model = train_random_forest(
    X_train,
    y_train,
    preprocessor,
    config
)
    xgb_model = train_xgboost(
    X_train,
    y_train,
    preprocessor,
    config
)
# nn_model = train_neural_network(
#     X_train,
#     y_train,
#     preprocessor,
#     config
# )
    best_model, results = compare_models(
    rf_model,
    xgb_model,
    X_test,
    y_test
)

    save_best_model(
    best_model,
    preprocessor
)

if __name__ == "__main__":
    main()