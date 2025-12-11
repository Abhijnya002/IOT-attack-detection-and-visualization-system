# IoT Attack Detection and Visualization System

## Project Overview
This project implements a comprehensive machine learning-based intrusion detection system for IoT networks using the CICIDS2017 dataset. The system evaluates six different machine learning models (Random Forest, Support Vector Machine, Neural Network, CNN, LSTM, and Autoencoder) to identify the most effective approach for detecting network attacks in IoT environments.

## Authors
- Ananya Naga Raj (annagara@syr.edu)
- Abhijnya Konanduru Gurumurthy (akonandu@syr.edu)
- Amulya Naga Raj (amnagara@syr.edu)

Syracuse University - College of Engineering and Computer Science
Graduate Research Project - Fall 2025

## Repository Structure

### Source Files (source.zip)
- **IOT_Security_dashboard_for_network_anomaly.ipynb** - Main Jupyter notebook containing:
  - Data preprocessing and cleaning
  - Feature engineering and label encoding
  - Training code for all 6 models
  - Model evaluation and performance metrics
  - SHAP explainability analysis
  - Visualization generation
  
- **iot_attack_detection_dashboard_UPDATED.html** - Interactive React-based dashboard featuring:
  - Executive summary with key findings
  - Complete model comparison tables
  - Detailed performance analysis for each model
  - Interactive confusion matrix visualization (click buttons to view)
  - SHAP feature importance with detailed explanations
  - Dataset distribution charts
  - Speed comparison visualizations
  - Production deployment recommendations
  
- **results.py** - Python script for results analysis and visualization generation

- **README.md** - This documentation file

### Output Files (output.zip)
- **random_forest_real.pkl** - Trained Random Forest model (99.90% accuracy, 10.2 MB)
- **svm_real.pkl** - Trained Support Vector Machine model (98.82% accuracy, 1.7 MB)
- **neural_network_real.h5** - Trained Neural Network model (97.52% accuracy, 114 KB)
- **cnn_model_real.h5** - Trained CNN model (95.84% accuracy, 711 KB)
- **lstm_model_real.h5** - Trained LSTM model (86.82% accuracy, 534 KB)
- **autoencoder_model_real.h5** - Trained Autoencoder model (65.98% accuracy, 81 KB)
- **autoencoder_threshold.pkl** - Anomaly detection threshold (0.7357) for Autoencoder
- **shap_values.pkl** - SHAP explainability values for Random Forest feature importance
- **project_summary_report.txt** - Summary of experimental results and key findings

## Dataset Information

### CICIDS2017 Dataset
The Canadian Institute for Cybersecurity Intrusion Detection System 2017 dataset provides realistic network traffic capturing both normal activity and various attack types.

**Dataset Statistics:**
- Total samples: 125,973 network flows
- Training set: 100,778 samples (80%)
- Test set: 25,195 samples (20%)
- Features: 38 numerical network traffic characteristics
- Classes: 4 categories (Normal, Other, R2L, U2R)

**Class Distribution (Test Set):**
- Normal: 13,469 samples (53.5%)
- Other (DoS, Probe): 11,706 samples (46.5%)
- R2L (Remote-to-Local): 12 samples (0.05%)
- U2R (User-to-Root): 8 samples (0.03%)

**Dataset Source:** https://www.unb.ca/cic/datasets/ids-2017.html

**Note:** The preprocessed dataset file (CICIDS2017_processed_UPDATED.csv, 16.8 MB) is not included in this submission due to file size constraints. The original dataset is publicly available at the link above.

## Attack Types Detected

### 1. Normal Traffic
Regular network communication without malicious activity.

### 2. DoS (Denial of Service)
Attacks that overwhelm system resources through SYN flooding, HTTP floods, or slowloris attacks. Detected through abnormal serror_rate and connection patterns.

### 3. Probe/Scan Attacks
Network reconnaissance activities including port scanning and service enumeration. Identified via dst_host_count and service connection patterns.

### 4. R2L (Remote-to-Local)
Unauthorized access attempts including brute force attacks and exploitation of vulnerable services. Detected through num_failed_logins and authentication patterns.

### 5. U2R (User-to-Root)
Privilege escalation attacks attempting to gain root/administrator access. Identified via num_shells, root_shell, and num_file_creations features.

## Model Performance Summary

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Inference Time | Status |
|------|-------|----------|-----------|--------|----------|----------------|--------|
| 1 | Random Forest | 99.90% | 99.90% | 99.90% | 99.90% | 0.726s | Production Ready |
| 2 | SVM | 98.82% | 99.12% | 98.82% | 98.96% | 22.894s | Backup Option |
| 3 | Neural Network | 97.52% | 98.94% | 97.52% | 98.20% | 1.491s | Research Only |
| 4 | CNN | 95.84% | 98.27% | 95.84% | 97.01% | 2.559s | Not Recommended |
| 5 | LSTM | 86.82% | 92.57% | 86.82% | 89.49% | 19.878s | Not Recommended |
| 6 | Autoencoder | 65.98% | 85.03% | 32.65% | 47.18% | 2.642s | Anomaly Detection Only |

### Key Findings:
- **Random Forest dominates** with highest accuracy, fastest inference, and minimal false positives
- **Traditional ML outperforms Deep Learning** by 31x (vs SVM) and 27x (vs LSTM) in speed
- **Deep learning struggles** with tabular IoT data lacking spatial/temporal structure
- **Class imbalance** (only 8 U2R samples) makes rare attack detection challenging
- **SHAP analysis** reveals security-relevant features matter more than volume metrics

## SHAP Explainability - Top Features

The SHAP analysis identifies the most important features for attack detection:

1. **serror_rate** (8.67%) - SYN error rate indicating DoS/DDoS attacks
2. **num_file_creations** (8.19%) - File system activity indicating intrusion
3. **num_shells** (7.73%) - Shell access attempts indicating privilege escalation
4. **srv_count** (6.71%) - Service connection patterns
5. **num_compromised** (2.85%) - Direct indicator of successful attacks
6. **dst_host_rerror_rate** (2.80%) - Rejection errors from scanning/probing
7. **urgent** (2.76%) - Urgent packets rare in normal traffic
8. **root_shell** (2.74%) - Root access attempts for U2R detection

These features provide interpretability and help security analysts understand model decisions.

## Technologies Used

### Machine Learning
- scikit-learn (Random Forest, SVM)
- TensorFlow/Keras (Neural Network, CNN, LSTM, Autoencoder)
- SHAP (Explainability)

### Data Processing
- pandas
- numpy
- StandardScaler

### Visualization
- matplotlib
- seaborn
- React (interactive dashboard)
- HTML/CSS/JavaScript

### Development Environment
- Google Colab / Jupyter Notebook
- Python 3.8+

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow shap matplotlib seaborn
```

### Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import shap
import matplotlib.pyplot as plt
import seaborn as sns
```

## Running the Project

### Step 1: Open Jupyter Notebook
1. Upload `IOT_Security_dashboard_for_network_anomaly.ipynb` to Google Colab or Jupyter
2. Upload the CICIDS2017 dataset (download from official source if needed)

### Step 2: Execute the Notebook
1. Run all cells sequentially from top to bottom
2. Data preprocessing will clean and prepare the dataset
3. All 6 models will be trained automatically
4. Evaluation metrics will be calculated
5. SHAP analysis will be performed on Random Forest
6. Model files will be saved to disk

### Step 3: View Results
1. Check the output cells for performance metrics
2. View confusion matrices for each model
3. Examine SHAP feature importance rankings
4. Open `iot_attack_detection_dashboard_UPDATED.html` in any web browser for interactive visualization

### Step 4: Load Pre-trained Models (Optional)
```python
import pickle
from tensorflow import keras

# Load Random Forest
with open('random_forest_real.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load Neural Network
nn_model = keras.models.load_model('neural_network_real.h5')

# Make predictions
predictions = rf_model.predict(X_test)
```

## Dashboard Usage

### Opening the Dashboard
1. Double-click `iot_attack_detection_dashboard_UPDATED.html`
2. Opens in your default web browser
3. No server or internet connection required - runs completely offline

### Dashboard Features
- **Overview Tab:** Executive summary, key metrics, model comparison table
- **Detailed Analysis Tab:** Individual model cards with metrics, clickable buttons to view confusion matrices
- **SHAP Explainability Tab:** Top 8 features with importance scores, clickable cards for detailed analysis
- **Dataset Info Tab:** CICIDS2017 statistics and class distribution
- **Conclusions Tab:** What works, what doesn't, deployment strategy, future work

### Interactive Elements
- Click "View Confusion Matrix" buttons on model cards to see detailed classification results
- Click "View Detailed Analysis" buttons on SHAP features to see importance visualizations
- Navigate between tabs to explore different aspects of the analysis

## Key Results

### Best Performing Model: Random Forest
- **Accuracy:** 99.90%
- **Inference Speed:** 0.726s for 25,195 samples (34,700 packets/second)
- **False Positives:** Very low (only 5 U2R false alarms)
- **U2R Detection:** 62.5% (5/8 attacks detected)
- **R2L Detection:** 91.67% (11/12 attacks detected)

### Why Traditional ML Wins
1. IoT network data is tabular without spatial/temporal patterns
2. Random Forest handles heterogeneous features effectively
3. No need for complex architectures when patterns are statistical
4. Faster inference critical for real-time IoT security
5. Better interpretability through feature importance

### Deep Learning Limitations
- **CNN:** Generated 603 false positives for U2R (wrong architecture for tabular data)
- **LSTM:** Generated 1,437 false positives (no temporal structure in data)
- **Both:** 10-100x slower inference compared to Random Forest

## SHAP Explainability Insights

### Why SHAP Matters
SHAP provides transparency into model decisions, showing exactly which features contributed to classifying traffic as malicious. This is critical for:
- Security analyst trust and validation
- Understanding attack patterns
- Debugging misclassifications
- Regulatory compliance and accountability

### SHAP vs Random Forest Feature Importance
- **SHAP Top Feature:** serror_rate (8.67%)
- **RF Top Feature:** dst_bytes (10.57%)
- **Difference:** SHAP shows actual prediction impact, RF shows splitting frequency
- **Insight:** Error rates and behavioral patterns matter more than volume metrics

## Challenges and Limitations

### 1. Extreme Class Imbalance
- Only 8 U2R and 12 R2L samples in test set (25,195 total)
- Even best models struggle with rare attack detection
- High class weights (787.33 for U2R) cause false positive explosions in neural networks

### 2. Deep Learning Unsuitability
- CNN and LSTM designed for spatial/sequential data
- IoT flow features lack meaningful ordering
- Models find spurious patterns that don't generalize

### 3. Real-time Deployment Constraints
- IoT devices have limited computational resources
- Models must balance accuracy with speed
- SVM and LSTM too slow despite acceptable accuracy

## Production Deployment Recommendations

### Primary System: Random Forest
- Deploy on edge IoT devices (low latency, small model size)
- Real-time classification at 34,700 packets/second
- SHAP explanations for security analyst review
- Minimal false positives (5 vs SVM's 81)

### Optional Enhancement: RF + SVM Ensemble
- If rare U2R detection is mission-critical
- SVM catches 1 additional U2R attack (6/8 vs RF's 5/8)
- Accept higher false positive rate (81 vs 1)
- Use weighted voting for final decision

## Future Work

### Immediate Improvements
1. **Synthetic Oversampling:** Apply SMOTE or ADASYN to R2L/U2R classes
2. **Ensemble Methods:** Weighted voting of RF + SVM for rare attack detection
3. **Hyperparameter Tuning:** Grid search for deep learning models with reduced class weights

### Long-term Enhancements
1. **Cross-dataset Validation:** Test on NSL-KDD, CSE-CIC-IDS2018
2. **Real-time Deployment:** Implement on Raspberry Pi or IoT gateway
3. **Adversarial Robustness:** Test against evasion attacks
4. **Online Learning:** Adaptive models that learn from new attack patterns
5. **Multi-stage Detection:** Combine anomaly detection (Autoencoder) with classification (Random Forest)

## Project Structure
```
IoT_Attack_Detection_Project/
├── source/
│   ├── IOT_Security_dashboard_for_network_anomaly.ipynb
│   ├── iot_attack_detection_dashboard_UPDATED.html
│   ├── results.py
│   └── README.md
├── output/
│   ├── random_forest_real.pkl
│   ├── svm_real.pkl
│   ├── neural_network_real.h5
│   ├── cnn_model_real.h5
│   ├── lstm_model_real.h5
│   ├── autoencoder_model_real.h5
│   ├── autoencoder_threshold.pkl
│   ├── shap_values.pkl
│   └── project_summary_report.txt
├── iot_attack_detection_system.pdf (Project Report)
└── presentation.pdf (Project Presentation)
```

## Technical Specifications

### Model Architectures

#### 1. Random Forest
- Trees: 100
- Max depth: 20
- Class weight: balanced
- Total parameters: Not applicable (ensemble)
- Training time: 7.28s

#### 2. Support Vector Machine
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- Gamma: scale
- Class weight: balanced
- Support vectors: 5,309
- Training time: 76.57s

#### 3. Neural Network
- Architecture: Dense layers (128 → 64 → 32 → 4)
- Activation: ReLU (hidden), Softmax (output)
- Regularization: Batch Normalization + Dropout (30%, 30%, 20%)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Total parameters: 16,356
- Epochs: 50
- Training time: ~200s

#### 4. Convolutional Neural Network
- Architecture: Conv1D (64, 128 filters) → Dense (128 → 64 → 4)
- Kernel size: 3
- Pooling: MaxPooling1D
- Regularization: Batch Normalization + Dropout
- Total parameters: 165,956
- Epochs: 50
- Training time: ~1200s

#### 5. LSTM
- Architecture: LSTM (128, 64 units) → Dense (64 → 32 → 4)
- Return sequences: True (first layer)
- Regularization: Batch Normalization + Dropout
- Total parameters: 123,108
- Epochs: 50
- Training time: ~6600s

#### 6. Autoencoder
- Encoder: 38 → 32 → 16 → 8 (bottleneck)
- Decoder: 8 → 16 → 32 → 38 (reconstruction)
- Loss: Mean Squared Error
- Threshold: 95th percentile (0.7357)
- Total parameters: 4,238
- Epochs: 50
- Training time: ~100s

### Feature Set (38 features)
Network traffic characteristics including:
- Flow duration and packet statistics
- Byte counts (source/destination)
- Error rates (serror, rerror)
- Connection patterns (same_srv_rate, dst_host_srv_count)
- Security indicators (num_failed_logins, num_shells, root_shell)
- Host statistics (dst_host_count, dst_host_diff_srv_rate)
- Service and flag information

## Detailed Results

### Confusion Matrix Performance

#### Random Forest (Best)
```
              Normal   Other   R2L   U2R
Normal        13,464      4     0     1
Other             17 11,689     0     0
R2L                1      0    11     0
U2R                3      0     0     5
```
- Clean classification with minimal errors
- Only 5 false positives total

#### SVM
```
              Normal   Other   R2L   U2R
Normal        13,297     94     9    69
Other            111 11,583     0    12
R2L                1      0    11     0
U2R                0      2     0     6
```
- 81 false positives (69 Normal → U2R, 12 Other → U2R)
- Detects 1 more U2R than RF (6/8 vs 5/8)

#### Neural Network
```
              Normal   Other   R2L   U2R
Normal        13,104    100    28   237
Other            144 11,449     7   106
R2L                0      1    11     0
U2R                1      0     0     7
```
- 343 false positives for U2R
- Best U2R recall among supervised models (7/8)

#### CNN
- 603 U2R false positives (443 Normal → U2R, 160 Other → U2R)
- Perfect U2R recall (8/8) but unusable due to false alarms
- Wrong architecture for tabular data

#### LSTM
- 1,437 U2R false positives (catastrophic)
- 86.82% accuracy (worst supervised model)
- Completely unsuitable for non-temporal data

#### Autoencoder
- Binary classification (Normal vs Attack)
- Misses 67% of attacks (only 32.65% recall)
- Good AUC (0.9236) but poor threshold-based detection

### Speed Analysis (25,195 samples)
- Random Forest: 0.726s (34,700 packets/sec) - **31x faster than SVM**
- Neural Network: 1.491s (16,900 packets/sec)
- CNN: 2.559s (9,800 packets/sec)
- Autoencoder: 2.642s (9,500 packets/sec)
- LSTM: 19.878s (1,300 packets/sec)
- SVM: 22.894s (1,100 packets/sec) - **Slowest**

## How to Use Pre-trained Models

### Load and Use Random Forest
```python
import pickle
import numpy as np

# Load model
with open('random_forest_real.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Prepare your data (must have same 38 features)
# X_new should be scaled using the same StandardScaler

# Make predictions
predictions = rf_model.predict(X_new)
probabilities = rf_model.predict_proba(X_new)

# Classes: 0=Normal, 1=Other, 2=R2L, 3=U2R
class_names = ['Normal', 'Other', 'R2L', 'U2R']
predicted_class = class_names[predictions[0]]
```

### Load and Use Neural Network
```python
from tensorflow import keras

# Load model
nn_model = keras.models.load_model('neural_network_real.h5')

# Make predictions
predictions = nn_model.predict(X_new)
predicted_class = np.argmax(predictions, axis=1)
```

### Load SHAP Values
```python
import pickle

# Load SHAP values
with open('shap_values.pkl', 'rb') as f:
    shap_values = pickle.load(f)

# Analyze feature importance
# Use for generating SHAP plots or understanding model decisions
```

## Reproducing Results

### Complete Workflow
1. Download CICIDS2017 dataset from official source
2. Open Jupyter notebook
3. Update dataset path in notebook
4. Run all cells sequentially:
   - Data loading and preprocessing
   - Feature engineering
   - Model training (6 models)
   - Evaluation and metrics calculation
   - SHAP analysis
   - Visualization generation
5. Models will be saved automatically
6. Open dashboard HTML to view interactive results

### Expected Runtime
- Data preprocessing: ~2-3 minutes
- Random Forest training: ~7 seconds
- SVM training: ~77 seconds
- Neural Network training: ~200 seconds (50 epochs)
- CNN training: ~1200 seconds (50 epochs)
- LSTM training: ~6600 seconds (50 epochs)
- Autoencoder training: ~100 seconds (50 epochs)
- SHAP analysis: ~1-2 minutes (500 samples)

**Total estimated time:** ~2-3 hours for complete execution

## Troubleshooting

### Common Issues

**Issue 1: Models not loading**
- Ensure you're using the same Python/library versions
- TensorFlow models (.h5) require compatible Keras version
- Use `pip install tensorflow==2.10.0` for compatibility

**Issue 2: SHAP analysis fails**
- SHAP requires specific versions: `pip install shap==0.41.0`
- Use smaller sample size if memory issues occur
- TreeExplainer works best with Random Forest

**Issue 3: Dashboard doesn't display**
- Ensure you open HTML file in a modern browser (Chrome, Firefox, Edge)
- Check browser console for JavaScript errors
- File must be opened locally (file://) not through a server

**Issue 4: Dataset path errors**
- Update file paths in notebook to match your directory structure
- Use absolute paths if relative paths fail
- Ensure CICIDS2017_processed_UPDATED.csv is in the same directory

## Performance Benchmarks

### Hardware Used
- Platform: Google Colab
- CPU: Intel Xeon (cloud instance)
- RAM: 12 GB
- GPU: Not used (CPU-only training)

### Scalability
- Dataset size: 125,973 samples processable in < 3 hours
- Random Forest: Scales linearly with data size
- Deep Learning: Scales quadratically (slower with more data)
- Recommended for: Up to 1M samples without optimization

## Citation

If you use this work, please cite:
```
A. Naga Raj, A. K. Gurumurthy, and A. Naga Raj, "IoT Attack Detection 
and Visualization System Using Machine Learning," Syracuse University, 
Graduate Research Project, Fall 2025.
```

## License
This project is submitted as academic coursework for Syracuse University. 
All code and documentation are provided for educational purposes.

## Contact
For questions or collaboration:
- Ananya Naga Raj: annagara@syr.edu
- Abhijnya Konanduru Gurumurthy: akonandu@syr.edu
- Amulya Naga Raj: amnagara@syr.edu

## Acknowledgments
- CICIDS2017 dataset provided by Canadian Institute for Cybersecurity
- SHAP library by Scott Lundberg
- Syracuse University College of Engineering and Computer Science
- Course Instructor: [Your Professor's Name]

## References
[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in Proceedings of the International Conference on Information Systems Security and Privacy (ICISSP), 2018, pp. 108-116.

[2] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.

[3] C. Cortes and V. Vapnik, "Support vector networks," Machine Learning, vol. 20, pp. 273-297, 1995.

[4] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[5] Y. LeCun et al., "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.

[6] S. Lundberg and S. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NIPS), 2017.

## Version History
- **v1.0** (November 2025): Initial release with 6 models, SHAP analysis, and interactive dashboard