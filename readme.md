# Software Review Intelligence Dashboard

A Python-based NLP project for **software review analysis**, **aspect-based summarization**, and **query-based review retrieval**.

This project allows users to:

* Load a `.jsonl` review dataset
* Automatically cluster reviews into software-related aspects
* Generate structured aspect summaries
* Search reviews using natural language queries
* Visualize aspect distributions and trends
* Export results from the GUI

---

# Features

## 1. Aspect-Based Review Analysis

The system groups reviews into the following aspects:

* Performance
* Installation
* Price
* Features
* Support
* Compatibility
* Interface
* Design

---

## 2. Structured Summarization

For each aspect, the system generates:

* Common theme
* Positive signal count
* Negative signal count
* Example review
* Top keywords

---

## 3. Semantic Search

Users can type a natural language query such as:

* `cheap software with easy installation`
* `software that crashes`
* `easy to use interface`

The system retrieves the most relevant review sentences.

---

## 4. GUI Dashboard

The Tkinter-based dashboard includes:

* Dataset loader
* Review analysis section
* Search query section
* Analysis results panel
* Search results panel
* Charts and export options

---

## 5. Charts

The dashboard supports:

* Aspect Distribution Chart
* Positive vs Negative Trend Chart

---

# Project Structure

```
software-review-dashboard/
│
├── app.py
├── config.py
├── review_processing.py
├── sentence_retrieval.py
├── summary_formatter.py
├── chart_utils.py
├── requirements.txt
├── readme.md
└── Software_5.jsonl
```

---

# Requirements

* Python **3.11 (recommended)**
* pip
* Internet connection (only required for first run to download models)

> ⚠️ Python 3.13 may cause issues with `torch` and `sentence-transformers`.
> Use **Python 3.11** for best compatibility.

---

# Installation Guide (Step-by-Step)

## Step 1: Download the Project

### Option A — Clone using Git

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### Option B — Download ZIP

* Click **Code → Download ZIP** on GitHub
* Extract the ZIP
* Open the folder in VS Code

---

## Step 2: Create Virtual Environment

### Windows

```
py -3.11 -m venv venv311
```

### Mac/Linux

```
python3.11 -m venv venv311
```

---

## Step 3: Activate Virtual Environment

### Windows (PowerShell)

```
venv311\Scripts\activate
```

### Windows (Command Prompt)

```
venv311\Scripts\activate.bat
```

### Mac/Linux

```
source venv311/bin/activate
```

You should now see:

```
(venv311)
```

---

## Step 4: Install Dependencies

```
pip install -r requirements.txt
```

---

## Step 5: Add Dataset

Place your dataset file inside the project folder:

```
Software_5.jsonl
```

Example structure:

```
software-review-dashboard/
├── app.py
├── Software_5.jsonl
└── ...
```

Example JSONL line:

```
{"overall": 5.0, "reviewText": "This software is very easy to install and use."}
```

---

# Running the Application

Run the GUI:

```
python app.py
```

This will launch the dashboard.

---

# How to Use

## 1. Load Dataset

* Click **Choose JSONL File**
* Select your dataset

## 2. Analyze Reviews

* Set review limit (recommended: `2000`)
* Click **Analyze Reviews**

## 3. Search Reviews

Enter a query such as:

```
cheap software with easy installation
```

Click **Search Reviews**.

## 4. View Charts

* Click **Aspect Chart**
* Click **Trend Chart**

## 5. Export Results

* Export analysis or search results as `.txt` files

---

# Example Queries

Try these:

```
cheap software
easy installation
software crashes often
good customer support
easy to use interface
works on mac
feature rich software
```

---

# Troubleshooting

## GUI freezes or slow

Reduce review count:

```
1000–2000 reviews recommended
```

---

## Module not found error

Run:

```
pip install -r requirements.txt
```

---

## Torch / transformer issues

Ensure Python version:

```
python --version
```

Use:

```
Python 3.11
```

---

## Hugging Face warning

```
Warning: unauthenticated requests
```

This is normal and safe to ignore.

---

# Notes

* First run is slower (model download)
* Later runs are faster (cached model)
* Designed for software review datasets

---

# Tech Stack

* Python
* Tkinter
* Sentence Transformers
* PyTorch
* Matplotlib

---

# Future Improvements

* Average rating per aspect
* Better summarization (LLM-based)
* Improved sentiment analysis
* Web-based dashboard

---

# Author

Your Name Here

---

# License

For academic and educational use.
