# ❒ IMDB Movie Review Sentiment Analysis using Simple RNN

This project implements a **binary sentiment analysis system** using a **Simple Recurrent Neural Network (SimpleRNN)** trained on the **IMDB Movie Reviews dataset**.  
It includes model training, evaluation, visualization of learning curves, and an interactive **Streamlit web application** for real-time sentiment prediction.

---

## ◧ Project Overview

Sentiment analysis is a fundamental task in Natural Language Processing (NLP).  
This project demonstrates how sequential models can be applied to text data to classify movie reviews as **Positive** or **Negative**.

The implementation follows an end-to-end workflow:
- Dataset loading and preprocessing
- Model training and validation
- Performance visualization
- Model persistence
- Interactive inference via Streamlit

---

## ◧ Problem Statement

Build a deep learning model that predicts the sentiment of a movie review based on its textual content.

### Target Variable
- `1` → Positive Review  
- `0` → Negative Review  

---

## ◧ Project Structure

```text
IMDB-Simple-RNN/
│
├── pro_1.py
│   # Model training, evaluation, and visualization script
│
├── app.py
│   # Streamlit application for real-time sentiment prediction
│
├── simple_rnn_imdb.h5
│   # Saved trained Simple RNN model
│
├── acc.png
│   # Training vs Validation Accuracy plot (generated during execution)
│
├── loss.png
│   # Training vs Validation Loss plot (generated during execution)
│
├── requirements.txt
│   # Project dependencies
│
└── README.md
    # Project documentation
```
---
## ◧ Tech Stack

### Programming Language
- **Python 3**

### Deep Learning & Machine Learning
- **TensorFlow / Keras**  

### Natural Language Processing
- **Keras IMDB Dataset**  

### Data Processing
- **NumPy**  
- **Pandas**  

### Visualization
- **Matplotlib**  

### Web Application
- **Streamlit**  

### Development Tools
- **VS Code**  
- **Git & GitHub**  

---
## ❒ Clone and Deploy the Project

- Step 1: Clone the Repository
```bash
git clone https://github.com/GitRzh/imdb-sentiment-analysis-rnn.git
cd imdb-sentiment-analysis-rnn
```
- Step 2: Create Virtual Enviroment
```bash
python -m venv venv
```
```bash
source venv/bin/activate        #linux/mac
```
```bash
venv\Scripts\activate           #windows
```
- Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
- Step 4: Run the Application Locally
```bash
streamlit run app.py
```

---

## ◧ Dataset

- IMDB Movie Reviews Dataset (Keras built-in)
- 50,000 movie reviews
- Binary sentiment labels: Positive (1), Negative (0)
- Vocabulary limited to top 1,000 most frequent words
- Reviews encoded as integer sequences

---

## ▣ Dataset Limitations

- Vocabulary size restriction leads to out-of-vocabulary tokens
- Integer encoding removes semantic meaning of rare words
- No explicit handling of sarcasm or implicit sentiment
- Fixed sequence length causes truncation of long reviews

---

## ▣ Model Limitations

- SimpleRNN suffers from vanishing gradient issues
- Limited ability to capture long-term dependencies
- Performs poorly on negation, sarcasm, and contextual sentiment
- Tends to bias predictions toward a dominant class
- Not suitable for complex or long textual inputs

---

## ▣ Future Improvements

- Replace SimpleRNN with LSTM to better retain long-term context
- Improve handling of negation and sequential dependencies
- Compare SimpleRNN and LSTM performance metrics
- Enhance model robustness for real-world text inputs

---

## © Made by

**Raz**

Python | AI & ML Enthusiast

---

## ✉ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.

Connect with Me!

**GitHub:** https://github.com/GitRzh

**E-mail:** GitRzh@users.noreply.github.com
