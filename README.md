
# IoT Attack Detection and Visualization System

## Overview
A machine-learning–based intrusion detection system for IoT networks using the **CICIDS2017** dataset. Six models (RF, SVM, NN, CNN, LSTM, Autoencoder) were trained and evaluated, with SHAP explainability and an interactive **React dashboard** for visualization.

## Authors
- **Ananya Naga Raj** 
- **Abhijnya Konanduru Gurumurthy** 
- **Amulya Naga Raj**

## Repository Structure

source/
IOT_Security_dashboard_for_network_anomaly.ipynb
iot_attack_detection_dashboard_UPDATED.html
results.py
output/
random_forest_real.pkl
svm_real.pkl
neural_network_real.h5
cnn_model_real.h5
lstm_model_real.h5
autoencoder_model_real.h5
shap_values.pkl
autoencoder_threshold.pkl
project_summary_report.txt
presentation.pdf

## Dataset (CICIDS2017)
- **125,973** flows, **38 features**, **4 classes**  
- Imbalanced: only 12 R2L and 8 U2R samples  
- Source: https://www.unb.ca/cic/datasets/ids-2017.html

## Attack Categories
- **Normal** – benign activity  
- **DoS/Other** – flooding, scanning  
- **Probe** – reconnaissance  
- **R2L** – unauthorized remote access  
- **U2R** – privilege escalation  

## Model Performance Summary
| Rank | Model | Accuracy | Speed | Notes |
|------|--------|----------|--------|-------|
| 1 | **Random Forest** | **99.90%** | **0.726s** | Best, deployable |
| 2 | SVM | 98.82% | 22.89s | Backup |
| 3 | Neural Network | 97.52% | 1.49s | Research only |
| 4 | CNN | 95.84% | 2.56s | High false positives |
| 5 | LSTM | 86.82% | 19.88s | Not suitable |
| 6 | Autoencoder | 65.98% | 2.64s | Anomaly-only |

### Key Findings
- Random Forest is **fastest + most accurate**  
- Deep learning performs poorly on **tabular IoT data**  
- Class imbalance severely affects rare attack detection  

## SHAP: Top Features
1. serror_rate  
2. num_file_creations  
3. num_shells  
4. srv_count  
5. num_compromised  
6. dst_host_rerror_rate  
7. urgent  
8. root_shell  

## Technologies Used
- **ML:** scikit-learn, TensorFlow/Keras  
- **Explainability:** SHAP  
- **Visualization:** React, HTML/CSS  
- **Data:** pandas, numpy, seaborn, matplotlib  

## How to Run
1. Open notebook: `IOT_Security_dashboard_for_network_anomaly.ipynb`  
2. Load CICIDS2017 dataset  
3. Run all cells to train and evaluate models  
4. Open `iot_attack_detection_dashboard_UPDATED.html` to view results  

### Load Pretrained RF Model
```python
import pickle
with open('random_forest_real.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(X_test)
````

## Dashboard Features

* Model comparison
* Confusion matrices
* SHAP feature importance
* Dataset stats
* Deployment recommendations

## Deployment Recommendation

* **Use Random Forest** for production
* Optional: RF + SVM ensemble for improved rare attack detection

## Future Work

* SMOTE/ADASYN oversampling
* Ensemble tuning
* Cross-dataset testing
* Edge deployment (Raspberry Pi)
* Adversarial robustness

```
