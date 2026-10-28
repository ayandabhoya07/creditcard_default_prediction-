# CreditStream: Real-Time Credit Default Prediction

An online machine learning application that simulates streaming financial data to predict credit defaults using **Adaptive Random Forest** and monitors model performance through an interactive Streamlit dashboard.

## Overview

CreditStream demonstrates how online machine learning can process incoming customer records, generate predictions, evaluate performance, and incrementally learn from new labeled data.

The application uses River's Adaptive Random Forest classifier and a Streamlit dashboard to visualize predictions, classification metrics, and customer risk information in a simulated streaming environment.

> **Note:** This project currently uses synthetic customer data. It is an educational and portfolio project, not a production credit-scoring system.

## Features

* **Streaming Data Simulation:** Continuously generates synthetic customer financial records.
* **Online Machine Learning:** Uses River's Adaptive Random Forest classifier.
* **Incremental Learning:** Updates the model as new labeled records arrive.
* **Prequential Evaluation:** Predicts first, evaluates the prediction, and then trains on the new record.
* **Live Performance Metrics:** Monitors accuracy, precision, recall, F1-score, and ROC-AUC.
* **Interactive Dashboard:** Displays model performance, a confusion matrix, and incoming customer records.
* **Applicant Risk Prediction:** Provides an interface to assess an individual applicant's profile.
* **CSV Export:** Allows generated records to be downloaded for further analysis.

## Technology Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** River — Adaptive Random Forest
* **Data Processing:** Pandas, NumPy
* **Synthetic Data Generation:** Faker
* **Visualization:** Plotly, where used

## How It Works

CreditStream follows a predict-then-learn workflow:

1. **Generate:** Create a synthetic customer financial record.
2. **Process:** Prepare the record's features for the model.
3. **Predict:** Estimate whether the customer will default.
4. **Evaluate:** Compare the prediction against the generated label.
5. **Learn:** Update the model incrementally using the labeled record.
6. **Monitor:** Display updated performance metrics and records on the dashboard.

### System Workflow

```text
Synthetic Customer Data
          |
          v
    Feature Processing
          |
          v
 Adaptive Random Forest
          |
          v
   Default Prediction
          |
          v
   Model Evaluation
          |
          v
 Incremental Learning
          |
          v
 Streamlit Dashboard
```

## Evaluation Metrics

| Metric           | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| Accuracy         | Percentage of correct predictions                            |
| Precision        | How many predicted defaults were actual defaults             |
| Recall           | How many actual defaults were correctly identified           |
| F1-score         | Harmonic mean of precision and recall                        |
| ROC-AUC          | Ability to distinguish between default and non-default cases |
| Confusion Matrix | Breakdown of correct and incorrect classifications           |

Credit default datasets can be imbalanced, so accuracy alone does not provide a complete picture of model performance. Precision, recall, F1-score, and ROC-AUC help provide a more balanced evaluation.

## Run Locally

### Prerequisites

* Python
* Git
* pip

### 1. Clone the repository

```bash
git clone https://github.com/ayandabhoya07/creditcard_default_prediction-.git
cd creditcard_default_prediction-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run streamlit_app.py
```

The dashboard should open at:

`http://localhost:8501`

## Limitations

* The current data stream is simulated rather than sourced from a live financial institution.
* Customer records and default outcomes are synthetically generated.
* Real-world default labels can be delayed, unlike the immediate labels used in this simulation.
* Model performance on synthetic data does not establish performance on real financial data.
* Production use would require real-world validation, security controls, monitoring, and appropriate governance.

## Future Improvements

Potential extensions include:

* Historical credit-risk data replay.
* Delayed-label evaluation.
* Explicit concept-drift experiments.
* Validation-based prediction threshold selection.
* Automated testing and continuous integration.
* Improved logging, observability, and deployment monitoring.

These are potential enhancements, not claims about features already implemented.

## Author

**Ayan Dabhoya**

* GitHub: [@ayandabhoya07](https://github.com/ayandabhoya07)
* Repository: [creditcard_default_prediction-](https://github.com/ayandabhoya07/creditcard_default_prediction-)

---

*CreditStream is an educational machine learning project and is not intended for real-world lending decisions or financial advice.*
