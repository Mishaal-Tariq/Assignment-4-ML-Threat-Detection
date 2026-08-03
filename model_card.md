# Model Card – Machine Learning Threat Detection

## Model Overview

This project implements a machine learning-based network intrusion detection system for identifying malicious and benign network traffic. Three machine learning algorithms were used during development: Random Forest, XGBoost, and a Neural Network (MLPClassifier). The best-performing model was selected and saved for deployment.

---

## Purpose

The purpose of this model is to classify network traffic as either benign or attack traffic using features extracted from network flow records. It is intended for educational purposes and demonstrates the complete machine learning workflow for cybersecurity threat detection.

---

## Dataset

- Dataset Type: Synthetic Network Traffic
- Total Records: 50,000
- Benign Records: 35,000
- Attack Records: 15,000
- Attack Categories:
  - DoS
  - PortScan
  - BruteForce

---

## Features

The model uses both numerical and categorical network features, including:

- Protocol Type
- Service
- Flag
- Duration
- Source Port
- Destination Port
- Forward Packets
- Backward Packets
- Total Packets
- Forward Bytes
- Backward Bytes
- Total Bytes
- Flow Bytes/s
- Flow Packets/s

---

## Algorithms Evaluated

- Random Forest
- XGBoost
- Multi-Layer Perceptron (Neural Network)

The best-performing model was automatically selected and saved after comparison.

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Results

The trained model achieved excellent classification performance on the evaluation dataset. The confusion matrix and performance metrics were generated and stored in the evaluation results.

---

## Limitations

- The dataset is synthetically generated and may not fully represent real-world network traffic.
- Performance on real enterprise environments may differ.
- Additional attack categories can be incorporated in future work.

---

## Future Improvements

- Train using real cybersecurity datasets such as CICIDS2017 or UNSW-NB15.
- Perform hyperparameter optimization on larger datasets.
- Add explainable AI techniques such as SHAP.
- Deploy the model as a real-time intrusion detection service.

---
## Performance Summary

The final selected model achieved the following evaluation results:

- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-Score: 100%

These results were obtained using the generated synthetic network traffic dataset.