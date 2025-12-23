# 🏥 Clinical Report Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

An advanced NLP pipeline designed to analyze radiology reports. This system combines rule-based NLP (NegBio) for precise disease extraction with Deep Learning (BERT) for context-aware medical question answering.

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🩺 About the Project

The **Clinical Report Analyzer** automates the extraction of critical information from unstructured medical text. It addresses two main challenges in clinical NLP:
1.  **Structured Information Extraction:** Identifying the presence or absence of specific pathologies while handling complex negations (e.g., *"No evidence of pneumonia"*).
2.  **Interactive Querying:** Allowing clinicians to ask natural language questions about patient reports (e.g., *"How old is the patient?"*) and retrieving precise answers.

This project was built using a hybrid approach, leveraging **NegBio** for rule-based disease detection and a fine-tuned **BERT** model for question answering.

---

## 🌟 Key Features

### 🔍 Disease Label Extraction
* Automatically detects **11 distinct pathologies**:
    * Cardiomegaly, Lung Lesion, Airspace Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture.
* Utilizes **Negation Detection** to distinguish between positive and negative findings.
* Provides visual highlights of disease keywords within the report.

### 💬 Medical Question Answering (Q&A)
* Powered by a fine-tuned **BERT** model (TensorFlow).
* Extracts exact answer spans from clinical passages based on user questions.
* Displays confidence metrics and processing time.

### 📊 Batch Processing
* Upload CSV files containing multiple reports.
* Process hundreds of reports in seconds.
* Export structured results (True/False labels) for downstream analysis.

### 🖥️ Interactive UI
* User-friendly web interface built with **Gradio**.
* Real-time visualizations and interactive charts.

---

## 📂 Project Structure

```text
Clinical-Report-Analyzer/
├── app.py                   # Main application entry point (Gradio UI)
├── setup_env.py             # Script to download models & dependencies
├── requirements.txt         # Python dependencies
├── src/
│   ├── preprocessing.py     # NLP logic: Text cleaning & NegBio
│   └── model.py             # Deep Learning logic: BERT loader & inference
├── data/                    # Folder for example datasets
└── README.md                # Project documentation

```

---

## 🛠 Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Hugging Face Transformers (BERT)
* **NLP:** NLTK, Bioc, NegBio (Rule-based negation)
* **Interface:** Gradio
* **Visualization:** Plotly
* **Data Manipulation:** Pandas, NumPy

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Clinical-Report-Analyzer.git](https://github.com/YourUsername/Clinical-Report-Analyzer.git)
cd Clinical-Report-Analyzer

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Setup Environment & Download Models:**
Run the setup script to clone the necessary NLP tools (NegBio) and download the pre-trained BERT model (~1.3GB) from Google Drive.
```bash
python setup_env.py

```



---

## 💡 Usage

### Running the Web App

To launch the interactive interface:

```bash
python app.py

```

After running the command, open your browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

### Using the Interface

1. **Tab 1 (Disease Extraction):** Paste a clinical report or select an example. Click "Analyze" to see detected diseases and their status.
2. **Tab 2 (Medical Q&A):** Enter a clinical passage and a question. The system will highlight the answer in the text.
3. **Tab 3 (Batch Processing):** Upload a CSV file with a `Report Impression` column. The system will process all rows and provide a downloadable CSV with predictions.

---

## 📊 Model Performance

The disease extraction pipeline was evaluated on a test set of **1,000 radiologist-labeled reports**.

| Metric | Score |
| --- | --- |
| **Average F1 Score** | **0.682** |
| Airspace Opacity F1 | 0.905 |
| Pleural Effusion F1 | 0.859 |
| Fracture F1 | 0.864 |

*Note: The system excels at detecting explicit findings but may require further fine-tuning for subtle or ambiguous cases.*

---

## 🙏 Acknowledgments

* **NegBio/ClinicalReport:** This project builds upon the open-source tools provided by the [ClinicalReport repository](https://www.google.com/search?q=https://github.com/ayushnangia/ClinicalReport).
* **BERT:** Google's BERT architecture via Hugging Face Transformers.
* **Dataset:** Evaluated using the Stanford Report Test dataset.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
