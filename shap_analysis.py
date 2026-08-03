import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
print("=" * 50)
print("SHAP Analysis")
print("=" * 50)

model = joblib.load(
    "model_artifacts/best_model.pkl"
)

print("Best model loaded successfully.")
df = pd.read_csv(
    "synthetic_data/synthetic_network_data_20260803_034339.csv"
)

print("Dataset loaded successfully.")
print(df.shape)
columns_to_drop = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "attack_type"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

X = df.drop("label", axis=1)

print("Data prepared for SHAP.")
preprocessor = joblib.load(
    "model_artifacts/preprocessor.pkl"
)

X_processed = preprocessor.transform(X)

print("Preprocessing completed.")
rf_model = model.named_steps["classifier"]

explainer = shap.TreeExplainer(rf_model)

print("SHAP explainer created.")
X_sample = X_processed[:500]

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated.")
os.makedirs("shap_analysis", exist_ok=True)

feature_names = preprocessor.get_feature_names_out()

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_analysis/summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Summary plot saved.")
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_analysis/bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Bar plot saved.")

