# Sentiment Classification of Amazon Handmade Product Reviews

## Overview

This project focuses on sentiment classification of online product reviews using traditional machine learning and deep learning methods.

The project compares TF-IDF and BERT-based text representations with different classification models, including SVM and BiLSTM.

## Models

The following models are implemented:

* TF-IDF + Linear SVM
* BERT + SVM
* BERT + BiLSTM
* BERT + BiLSTM + Attention + CLS Fusion

## Dataset

The project uses the **Handmade Products** subset of the **Amazon Reviews 2023** dataset.

Dataset source:

https://amazon-reviews-2023.github.io/

Sentiment labels are generated from review ratings:

* Rating > 3: Positive
* Rating = 3: Neutral
* Rating < 3: Negative

The full dataset is not included in this repository because of its large size.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* scikit-learn
* pandas
* NumPy
* Matplotlib

## Installation

Install the required packages:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib tqdm
```

## Author

**Huanzhang Xiang**
MSc Advanced Computer Science
University of Leeds
