# 🤖 Text Summarization using PEGASUS

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This project implements an **abstractive text summarization system** using Google's **PEGASUS transformer architecture** and the **SAMSum dialogue summarization dataset**.

The objective is to build a deep learning model capable of reading conversational dialogues and generating concise, meaningful summaries while preserving the most important information.

Unlike extractive summarization, which selects existing sentences from a document, abstractive summarization generates new text using a sequence-to-sequence Transformer model.

### 🎯 Main Objective

The project focuses on developing an end-to-end NLP pipeline that:

* Loads and preprocesses the SAMSum dataset
* Tokenizes dialogue and summary text
* Fine-tunes a pretrained PEGASUS model
* Evaluates the model using ROUGE metrics
* Generates summaries for unseen dialogues
* Demonstrates practical deployment of a Transformer-based NLP model

---

## 🧠 Model Architecture

The project uses:

**PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization)**

PEGASUS is a Transformer-based sequence-to-sequence model specifically designed for abstractive summarization tasks.

### Model Pipeline

```text
                 Input Dialogue
                       │
                       ▼
              Text Preprocessing
                       │
                       ▼
                 PEGASUS Tokenizer
                       │
                       ▼
              Encoder / Transformer
                       │
                       ▼
                   Decoder
                       │
                       ▼
             Generated Summary
                       │
                       ▼
                ROUGE Evaluation
```

---

## 📊 Dataset

### SAMSum Dataset

The **SAMSum dataset** contains realistic conversations together with human-written summaries.

Example:

**Dialogue**

```text
John: Are you coming to the meeting?
Sarah: Yes, I'll be there at 10.
John: Great. See you then!
```

**Reference Summary**

```text
Sarah confirms she will attend the meeting at 10.
```

The dataset is particularly useful for training models to summarize:

* Chat conversations
* Messages
* Informal discussions
* Multi-speaker dialogues

---

## ✨ Key Features

* 🔤 Natural Language Processing
* 🤖 Transformer-based deep learning
* 🧠 PEGASUS pretrained model
* 💬 Dialogue summarization
* 📚 SAMSum dataset
* 🎯 Abstractive text generation
* 📏 ROUGE-based evaluation
* 🚀 GPU acceleration
* 🧪 Model evaluation
* 📝 Custom summary generation

---

# 🛠️ Technologies Used

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| Python                    | Programming language            |
| PyTorch                   | Deep learning framework         |
| Hugging Face Transformers | PEGASUS implementation          |
| Hugging Face Datasets     | Dataset loading and processing  |
| SentencePiece             | Tokenization                    |
| Evaluate                  | Model evaluation                |
| ROUGE                     | Summarization evaluation        |
| Google Colab              | Development and GPU environment |
| Jupyter Notebook          | Experimentation                 |

---

# 📁 Project Structure

```text
text-summarization-pegasus/
│
├── README.md
│
├── notebooks/
│   └── PEGASUS_Text_Summarization.ipynb
│
├── requirements.txt
│
└── .gitignore
```

> The exact folder structure can be modified depending on how the project is organized in your GitHub repository.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/text-summarization-pegasus.git
```

Move into the project directory:

```bash
cd text-summarization-pegasus
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
transformers
datasets
evaluate
rouge_score
sentencepiece
torch
accelerate
numpy
pandas
tqdm
```

---

# 🚀 Running the Project

## Option 1 — Google Colab

The recommended way to run this project is using Google Colab with GPU acceleration.

Open the notebook:

```text
notebooks/PEGASUS_Text_Summarization.ipynb
```

Then:

1. Open the notebook in Google Colab.
2. Enable GPU.
3. Install the required libraries.
4. Load the SAMSum dataset.
5. Preprocess the dataset.
6. Tokenize the dialogues.
7. Fine-tune PEGASUS.
8. Evaluate the model.
9. Generate summaries.

---

# 🔧 Hardware Requirements

Training Transformer models can be computationally expensive.

### Recommended

```text
GPU: NVIDIA T4 / RTX 3060 or better
RAM: 12 GB+
Storage: 10 GB+
Python: 3.9+
```

Google Colab GPU is sufficient for experimentation and educational purposes.

---

# 📚 Methodology

The project follows the following machine learning pipeline:

```text
Dataset
   │
   ▼
Data Exploration
   │
   ▼
Data Preprocessing
   │
   ▼
Tokenization
   │
   ▼
Train / Validation / Test
   │
   ▼
PEGASUS Fine-Tuning
   │
   ▼
Model Evaluation
   │
   ▼
ROUGE Metrics
   │
   ▼
Summary Generation
```

---

# 1️⃣ Environment Setup

The first stage installs and imports the required libraries.

Main libraries include:

```python
import torch
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    TrainingArguments,
    Trainer
)
```

The availability of a GPU is also checked before training.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
```

---

# 2️⃣ Dataset Loading

The SAMSum dataset is loaded using Hugging Face Datasets.

```python
from datasets import load_dataset

dataset = load_dataset("samsum")
```

The dataset contains:

```text
Dialogue
Summary
```

and is divided into training, validation, and test sets.

---

# 3️⃣ Data Preprocessing

The dialogue and summary data are cleaned and prepared for the Transformer model.

The input is:

```text
Dialogue
```

The target is:

```text
Summary
```

Special tokens and unnecessary formatting are handled during preprocessing.

---

# 4️⃣ Tokenization

PEGASUS requires tokenized input.

The pretrained tokenizer is loaded using:

```python
from transformers import PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained(
    "google/pegasus-cnn_dailymail"
)
```

The dialogue is converted into numerical token IDs that can be processed by the Transformer architecture.

---

# 5️⃣ Model Initialization

The pretrained PEGASUS model is loaded using:

```python
from transformers import PegasusForConditionalGeneration

model = PegasusForConditionalGeneration.from_pretrained(
    "google/pegasus-cnn_dailymail"
)
```

The model is then fine-tuned using the SAMSum dataset.

---

# 6️⃣ Model Training

During training, PEGASUS learns the relationship between:

```text
Dialogue → Summary
```

The model updates its parameters using the training data to minimize the sequence-to-sequence loss.

A GPU is used whenever available to accelerate training.

Example training configuration:

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)
```

---

# 7️⃣ Model Evaluation

The trained model is evaluated on unseen test data.

The primary evaluation metrics are:

* ROUGE-1
* ROUGE-2
* ROUGE-L

These metrics compare the generated summaries against the reference summaries.

---

# 📏 ROUGE Metrics

## ROUGE-1

Measures the overlap of individual words between the generated summary and reference summary.

## ROUGE-2

Measures the overlap of two-word sequences, known as bigrams.

## ROUGE-L

Measures the longest common subsequence between the generated and reference summaries.

Higher ROUGE scores generally indicate greater similarity to the reference summaries.

---

# 📊 Evaluation Results

The model evaluation results should be reported here after running the notebook.

Example format:

| Metric  |     Score |
| ------- | --------: |
| ROUGE-1 | **0.020023** |
| ROUGE-2 | **0.000000** |
| ROUGE-L | **0.019792** |


### Example Result Interpretation

```text
ROUGE-1: Measures unigram overlap
ROUGE-2: Measures bigram overlap
ROUGE-L: Measures sentence-level sequence similarity
```

The results provide an indication of how effectively the fine-tuned PEGASUS model summarizes unseen conversations.

---

# 🧪 Example Prediction

After training, the model can generate summaries for new dialogues.

### Input

```text
Alice: Did you finish the project?
Bob: Almost. I just need to complete the final report.
Alice: Great. Please send it to me when you're done.
Bob: Sure, I'll send it this afternoon.
```

### Generated Summary

```text
Bob is finishing the project report and will send it to Alice this afternoon.
```

The model generates the summary automatically based on the dialogue context.

---

# 🔮 Custom Prediction

The trained model can be used to summarize new conversations.

Example:

```python
dialogue = """
Alice: Are you available for the meeting tomorrow?
Bob: Yes, I am available at 10 AM.
Alice: Perfect. Let's meet at 10.
"""

inputs = tokenizer(
    dialogue,
    return_tensors="pt",
    truncation=True
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=60,
    min_length=10,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print(summary)
```

---

# 📈 Performance Analysis

The performance of the model depends on several factors:

* Dataset size
* Training epochs
* Learning rate
* Batch size
* Maximum input length
* Maximum output length
* GPU availability
* Beam search configuration
* Quality of preprocessing

Fine-tuning for additional epochs may improve performance, but it can also increase training time and potentially cause overfitting.

---

# ⚠️ Limitations

Although PEGASUS performs well for abstractive summarization, the project has several limitations.

### 1. Computational Requirements

Transformer models require significant computational resources for training.

### 2. Input Length

Very long conversations may need to be truncated because Transformer models have maximum input lengths.

### 3. Hallucination

The model may occasionally generate information that is not explicitly present in the original dialogue.

### 4. Training Time

Fine-tuning a large Transformer model can take considerable time without GPU acceleration.

### 5. Domain Dependency

Performance may decrease when summarizing conversations that are significantly different from the training dataset.

---

# 🔐 Ethical Considerations

Automated summarization systems should be used carefully when processing sensitive conversations.

Potential concerns include:

* Privacy
* Data security
* Incorrect summaries
* Hallucinated information
* Bias in generated summaries

Sensitive conversations should not be processed without appropriate authorization and data protection measures.

---

# 🚀 Future Improvements

Several improvements can be made to this project.

### 🔹 Model Improvements

* Experiment with larger PEGASUS models
* Compare PEGASUS with T5
* Compare with BART
* Experiment with modern instruction-tuned models

### 🔹 Training Improvements

* Increase training epochs
* Perform hyperparameter tuning
* Use learning-rate scheduling
* Experiment with different batch sizes
* Apply gradient accumulation

### 🔹 Evaluation Improvements

Add additional metrics such as:

* BERTScore
* BLEURT
* METEOR

Human evaluation can also be introduced to measure:

* Fluency
* Relevance
* Factual consistency
* Readability

### 🔹 Deployment

The model could be deployed using:

```text
FastAPI
      ↓
REST API
      ↓
PEGASUS Model
      ↓
Generated Summary
```

A web application could also be developed using:

```text
React / HTML / CSS
        ↓
FastAPI
        ↓
PEGASUS
        ↓
Summary
```

---

# 🌐 Possible Real-World Applications

This project can be extended to applications such as:

* 💬 Chat summarization
* 📧 Email summarization
* 📝 Meeting summarization
* 🎧 Customer-support conversation summaries
* 📞 Call-center summaries
* 📱 Messaging application summaries
* 🏢 Business meeting reports
* 📚 Document summarization

---

# 🎓 Learning Outcomes

Through this project, the following concepts are demonstrated:

### Machine Learning

* Dataset preparation
* Training and validation
* Model evaluation
* Prediction

### Deep Learning

* Neural networks
* Transformer architecture
* Sequence-to-sequence learning
* Fine-tuning pretrained models

### NLP

* Text preprocessing
* Tokenization
* Abstractive summarization
* Text generation

### MLOps / Engineering

* Environment management
* GPU acceleration
* Model saving
* Evaluation pipelines
* Reproducible experiments
* GitHub project organization

---

# 🧩 Challenges Faced

During development, several challenges may occur:

### Dataset and preprocessing

Different datasets require different preprocessing strategies.

### Transformer memory usage

PEGASUS can require substantial GPU memory, especially with large batch sizes.

### Evaluation libraries

Different versions of the Hugging Face evaluation ecosystem may require different configurations.

### Training performance

Training speed depends heavily on available hardware.

---

# 🛠️ Troubleshooting

## CUDA / GPU Not Available

Check:

```python
torch.cuda.is_available()
```

If it returns:

```text
False
```

the notebook is running on CPU.

In Google Colab, enable:

```text
Runtime → Change runtime type → GPU
```

---

## Out of Memory Error

Reduce the batch size:

```python
per_device_train_batch_size=2
```

or:

```python
per_device_train_batch_size=1
```

You can also reduce the maximum input sequence length.

---

## Metric Loading Error

If an evaluation metric fails to load, make sure the required packages are installed:

```bash
pip install evaluate rouge_score
```

Then restart the runtime if necessary.

---

# 📦 Requirements

Recommended Python version:

```text
Python 3.9+
```

Required packages:

```text
torch
transformers
datasets
evaluate
rouge_score
sentencepiece
accelerate
numpy
pandas
tqdm
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

# 📜 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project according to the terms of the license.

---

# 👨‍💻 Author

**Sudeera Attanayake**

AI Engineer

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Natural Language Processing
* Agentic AI
* Generative AI
* MLOps

---

# ⭐ Acknowledgements

Special thanks to the open-source communities behind:

* Hugging Face
* PyTorch
* Google Research
* SAMSum dataset contributors
* Google Colab

This project was developed for educational and portfolio purposes.

---

# 📚 References

* PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
* SAMSum: A Human-annotated Dialogue Dataset for Abstractive Summarization
* Hugging Face Transformers documentation
* Hugging Face Datasets documentation
* ROUGE evaluation methodology

---

# ⭐ Project Highlights

```text
┌─────────────────────────────────────────────┐
│       PEGASUS TEXT SUMMARIZATION            │
├─────────────────────────────────────────────┤
│                                             │
│  Dataset       → SAMSum                     │
│  Architecture  → Transformer / PEGASUS      │
│  Task          → Abstractive Summarization  │
│  Framework     → PyTorch                    │
│  NLP Library   → Hugging Face Transformers  │
│  Evaluation    → ROUGE                     │
│  Environment   → Google Colab / Python      │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 📌 Conclusion

This project demonstrates the complete development workflow for an **abstractive dialogue summarization system using PEGASUS**.

By fine-tuning a pretrained Transformer model on the SAMSum dataset, the system learns to transform lengthy conversational text into concise summaries.

The project provides practical experience with:

```text
NLP
 ↓
Dataset Processing
 ↓
Tokenization
 ↓
Transformer Models
 ↓
Fine-Tuning
 ↓
Evaluation
 ↓
Text Generation
 ↓
Deployment Possibilities
```

The project can serve as a foundation for developing more advanced NLP applications such as meeting summarization, customer-support summarization, email summarization, and intelligent conversational assistants.

---

## ⭐ If you found this project useful

Give this repository a ⭐ on GitHub and feel free to fork the project and experiment with different Transformer architectures and datasets.

