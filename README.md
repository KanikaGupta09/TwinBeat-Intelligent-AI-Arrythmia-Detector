# 🫀 TwinBeat
<img width="900" height="220" alt="image" src="https://github.com/user-attachments/assets/c09f1ce4-9c24-4374-9137-d03fa238c75b" />


### AI-Powered Arrhythmia Detection & Digital Twin Simulator (ML-based)

TwinBeat is a **machine-learning–driven healthcare analytics project** that focuses on **early detection of cardiac arrhythmias using ECG signals**, along with a **digital twin comparison** to improve interpretability for doctors and patients.

The project uses **classical ML techniques (no deep learning, no NLP)** and is built on **clinically validated ECG datasets**.

### Here's my Published Research paper in Journal of Applied Bio Analysis : https://journalofappliedbioanalysis.com/comparative-analysis-of-machine-learning-techniques-for-arrhythmia-detection

---

## 📌 Problem Statement

Arrhythmia is often referred to as a *silent cardiac disorder* because:

* It can be **intermittent** and missed during routine checkups
* Doctors usually require **multiple ECG recordings** to confirm abnormalities
* Early-stage arrhythmias may not show obvious symptoms

This project aims to **assist clinicians** by using machine learning to:

* Detect arrhythmia patterns from ECG signals
* Highlight deviations from a healthy heart using a digital twin concept
* Provide an interpretable and explainable pipeline

---


## 🔁 Project Flow & Machine Learning Lifecycle

This project follows a structured end-to-end machine learning pipeline, starting from raw ECG data and ending with a validated arrhythmia prediction model.

### 1️⃣ Data Acquisition

Use clinically validated ECG datasets (PTB-XL primarily).

Import:

* ptbxl_database.csv for metadata and labels
* scp_statements.csv for diagnosis mapping
* Raw ECG signals from .dat + .hea files using WFDB

### 2️⃣ Data Understanding & Exploration

* Understand ECG structure (12-lead, sampling rate, duration).
* Study label distribution (normal vs arrhythmia cases).
* Identify rhythm-based arrhythmia labels (AFIB, PVC, PAC, etc.).
* Check signal quality, noise flags, and demographic coverage.

### 3️⃣ Data Cleaning & Preparation

* Convert diagnostic labels from encoded format to usable targets.
* Filter ECG records relevant to arrhythmia prediction.
* Handle noisy or low-quality signals where required.
* Normalize ECG signal amplitudes for consistency.

### 4️⃣ Signal Processing

Read raw ECG waveforms using .dat + .hea files.

Apply:

* Baseline drift removal
* Optional band-pass filtering
* Segment ECG into meaningful windows if required.

### 5️⃣ Feature Engineering (Core ML Step)

* Since classical ML models cannot directly learn from raw waveforms:
* Extract time-domain features:
* Heart rate
* RR intervals
* Signal variance, mean, skewness
* Extract morphological features:
* QRS duration
* Peak amplitudes
* Extract frequency-domain features:
* FFT or wavelet-based energy features

Result: A structured tabular dataset where

* Rows = ECG records / segments
* Columns = engineered features
* Target = arrhythmia label

### 6️⃣ Dataset Construction

* Combine features, labels, and metadata into a single ML-ready table.
* Perform train–test split with stratification to handle class imbalance.
* Ensure no patient-level data leakage across splits.

### 7️⃣ Model Training

* Train and compare multiple classical ML models:
* Logistic Regression (baseline)
* Random Forest
* Support Vector Machine (SVM)
* Decision Tree
* Ensemble Voting Classifier
* Hyperparameters are tuned to optimize performance.

### 8️⃣ Model Evaluation

Evaluate models using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Focus is placed on recall and sensitivity, which are critical for medical risk detection.

### 9️⃣ Interpretation & Explainability

* Analyze feature importance (especially for tree-based models).
* Understand which ECG features contribute most to arrhythmia detection.
* Ensure predictions are explainable for clinical relevance.

### 🔟 Deployment Readiness

* Serialize trained ML models.
* Prepare inference pipeline for integration into a web application.
* Ensure consistent preprocessing between training and prediction.
* This structured lifecycle ensures the project is:
* Clinically meaningful
* ML-correct
* Explainable
* Scalable for real-world use

## 📊 Dataset Used

### Primary Dataset: **PTB-XL ECG Dataset**

* 21,000+ clinical ECG recordings
* 12-lead ECGs
* 10-second duration per record
* Labels include rhythm disorders (AFIB, PVC, PAC, etc.)

**Files used:**

* `ptbxl_database.csv` → metadata & labels
* `scp_statements.csv` → label definitions
* `.dat` + `.hea` → raw ECG waveform data (via WFDB)

---

## 🧬 Raw ECG Signal (What the Data Looks Like)

![Image](https://www.researchgate.net/publication/330107119/figure/fig4/AS%3A710964617965569%401546518582166/lead-ECG-waveform-in-10-minutes-after-coronary-occlusion-A-control-model-without.png)

![Image](https://www.researchgate.net/publication/327263145/figure/fig1/AS%3A664565545193472%401535456181051/Waves-of-a-lead-II-ECG.png)

![Image](https://www.physio-pedia.com/images/thumb/a/a3/Arrhythmias.jpeg/439px-Arrhythmias.jpeg)

Each ECG record contains:

* **12 leads** (I, II, III, aVR, aVL, aVF, V1–V6)
* **1000 samples at 100 Hz** (or 500 Hz for high-resolution data)
* Stored in binary `.dat` format with metadata in `.hea`

---

## 🔄 Data Processing Pipeline

![Image](https://www.researchgate.net/publication/351472043/figure/fig3/AS%3A11431281102418393%401669457046729/Signal-processing-pipeline-Block-diagram-of-signal-processing-overview-showing-signal.ppm)

![Image](https://www.researchgate.net/publication/368972862/figure/fig3/AS%3A11431281125380967%401678289964984/Pipeline-for-feature-extraction-and-validation.png)

![Image](https://www.researchgate.net/publication/383823525/figure/fig1/AS%3A11431281276453767%401725653057227/Block-Diagram-Of-The-Workflow-Of-The-Proposed-ECG-Based-Arrhythmia-Detection-Scheme-31.ppm)

### Step-by-step flow:

1. **Metadata Loading**

   * Read `ptbxl_database.csv`
   * Parse `scp_codes` into usable labels

2. **Label Filtering**

   * Extract **rhythm-based arrhythmia cases**
   * Convert into binary or multiclass targets

3. **Signal Reading**

   * Load `.dat` + `.hea` using `wfdb.rdrecord()`
   * Extract ECG waveform as NumPy arrays

4. **Signal Cleaning**

   * Baseline drift removal
   * Normalization
   * Optional band-pass filtering

5. **Feature Engineering (for ML)**

   * Heart rate & RR intervals
   * Statistical features (mean, variance, skewness)
   * Morphological features (QRS width, peak amplitude)
   * Frequency-domain features (FFT / wavelets)

6. **Final ML Table**

   * One row = one ECG record / segment
   * Columns = extracted features
   * Target = arrhythmia label

---

## 🤖 Machine Learning Models Used

This project **does NOT use deep learning**.

### ML Models Trained:

* **Logistic Regression** (baseline)
* **Random Forest Classifier**
* **Support Vector Machine (SVM)**
* **Decision Tree**
* **Ensemble Voting Classifier**

### Why Classical ML?

* More **interpretable** for healthcare
* Requires **less data**
* Easier to explain predictions to doctors
* Suitable for structured ECG features

---

## 🧠 Digital Twin Concept

![Image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-023-34098-8/MediaObjects/41598_2023_34098_Fig1_HTML.png)

![Image](https://cdn.prod.website-files.com/60b553829ae790f1371bfcf1/64c931e9ae9591e0806fa1c9_Sinus%20Rhythm%20Vs.%20Sinus%20Arrhythmia%20.webp)

![Image](https://img.washingtonpost.com/wp-apps/imrs.php?high_res=true\&src=https%3A%2F%2Farc-anglerfish-washpost-prod-washpost.s3.amazonaws.com%2Fpublic%2FMRDUI3C3SBBIJANM6BA2DGA3TQ.png\&w=2048)

The digital twin module:

* Builds a **healthy ECG reference profile**
* Compares patient ECG vs healthy baseline
* Highlights deviations in rhythm, amplitude, and intervals

This helps users visually understand:

> “How different is my heart from a normal one?”

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Class imbalance is handled using:

* Stratified splits
* Class weighting
* Feature selection

---

## 🌐 Web Application (Planned Integration)

The ML model will be integrated into a web app that provides:

* ECG upload interface
* Real-time arrhythmia prediction
* ECG waveform visualization
* Digital twin comparison
* Medical chatbot & voice assistant for explanations

*(Web development handled separately from ML pipeline)*

---

## 🛠️ Tech Stack

**Data & ML**

* Python
* Pandas, NumPy
* scikit-learn
* WFDB
* SciPy

**Visualization**

* Matplotlib
* Plotly

**Deployment (Planned)**

* Flask / FastAPI backend
* React frontend

---

## 📚 Research-Backed Approach

This project is inspired by classical ML research from:

* MIT-BIH Arrhythmia studies
* PTB-XL benchmarking papers
* Random Forest & HRV-based arrhythmia detection literature

---

## 👥 Team Structure

* **ML & Analytics**: ECG preprocessing, feature engineering, modeling
* **Web Development**: UI, visualization, API integration

All team members participate in **cross-learning**.

---

## 🚀 Future Enhancements

* Multiclass arrhythmia detection
* Longitudinal ECG trend analysis
* Wearable ECG integration
* Doctor feedback loop (human-in-the-loop ML)

---

## 📌 Disclaimer

This project is **for educational and research purposes only** and does not replace professional medical diagnosis.



Just tell me 👍
