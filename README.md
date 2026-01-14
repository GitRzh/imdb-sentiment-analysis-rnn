# 📊 IMDB Movie Review Sentiment Analysis using Simple RNN

Predict whether a movie review expresses **positive** or **negative** sentiment using a **Simple Recurrent Neural Network (RNN)**.  
This project demonstrates an **end-to-end Deep Learning workflow** — data loading, preprocessing, model training, evaluation, and visualization.

---

## 📖 Project Overview

Sentiment analysis is a core Natural Language Processing (NLP) task used in recommendation systems, opinion mining, and social media analysis.  
This project uses the **IMDB Movie Reviews dataset** to classify reviews as positive or negative based on textual patterns learned by an RNN.

### Key Highlights
- End-to-end Deep Learning project
- Simple RNN model built using TensorFlow / Keras
- Text preprocessing using padding and word indexing
- Training visualization with accuracy and loss graphs
- Clean and minimal project structure

---

## 🧠 Problem Statement

Build a deep learning model that predicts sentiment from movie reviews based on:
- Word sequences
- Contextual dependencies in text
- Learned word embeddings

### Target Variable
- `1` → Positive Review  
- `0` → Negative Review  

---

## 🏗️ Project Structure

```text
IMDB-Simple-RNN/
│
├── pro_1.py                  # Model training, evaluation, and visualization
├── simple_rnn_imdb.h5        # Saved trained RNN model
└── README.md                 # Project documentation
```
---
## ▶️ How to Run the Project
```bash
Step 1: Clone the Repository
git clone <your-repo-url>
cd IMDB-Simple-RNN

Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

Step 3: Install Dependencies
pip install tensorflow numpy matplotlib

Step 4: Run the Script
python pro_1.py
```
---

---
## 👤 Author

**Raz**

Python | AI & ML Enthusiast

---

### ⭐ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.

Connect with Me!

**GitHub:** https://github.com/GitRzh

**E-mail:** GitRzh@users.noreply.github.com