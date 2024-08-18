# Model Performance Comparison

## Background

This project involves training three variants of large language models (LLMs) to predict email subject lines given the email body. The models used are:

- **Mistral-7B**
- **LLaMA-8B**
- **BART**
- **Gemma-7B**

The training was performed using the [AESLC](https://github.com/ryanzhumich/AESLC) dataset. This dataset provides a corpus for email subject line prediction, enabling us to fine-tune pre-trained models to improve their performance on this specific task.

## Training Details

The models were trained and fine-tuned using the provided Python code. The training process involved:

1. **Data Preprocessing**: The AESLC dataset was prepared by extracting the email bodies and their corresponding subject lines.
2. **Fine-Tuning**: Each pre-trained model was fine-tuned on the AESLC dataset using custom training scripts.
3. **Evaluation**: The performance of the models was evaluated using the ROUGE metric, focusing on precision, recall, and F-measure.

## Performance Comparison

The performance of the models was evaluated in two scenarios:
1. **Pre-trained Models with Fine-Tuning**: The pre-trained models were fine-tuned using model generator parameters
2. **Models Trained Using AESLC Corpus**: The models were trained from scratch using the AESLC corpus.

The results are summarized in the table below:

| Model      | Training Method                  | ROUGE Precision | ROUGE Recall | ROUGE F-measure |
|------------|----------------------------------|-----------------|--------------|-----------------|
| Mistral-7B | Pre-trained with Fine-Tuning     | 0.8837          | 0.0478       | 0.0897          |
| Mistral-7B | Trained Using AESLC Corpus       | 0.8925          | 0.0455       | 0.0854          |
| LLaMA-8B   | Pre-trained (without fine-tuning)| 0.2645          | 0.2120       | 0.2133          |
| LLaMA-8B   | Trained Using AESLC Corpus       | 0.325           | 0.3166       | 0.3142          |
| BART       | Pre-trained with Fine-tuning     | ****            | *****        | ******          |
| BART       | Trained with AESLC               | 0.1350          | 0.0803       | 0.0919          |
| Gemma-7B   | Pre-trained                      | 0.2625          | 0.2167       | 0.2314          |
| Gemma-7B   | Trained Using AESLC Corpus       | 0.2910          | 0.4127       | 0.3116          |


## Python Training Code

The Python code used for training and fine-tuning the models is available in this repository. The main scripts include:

- `preprocess_data.py`: Script for preprocessing the AESLC dataset.
- `train_model.py`: Script for training the models.
- `evaluate_model.py`: Script for evaluating the model performance.


# Model Performance for Subject Line Generation

This document provides a comparative analysis of three models used for generating subject lines: **Mistral-7B**, **LLaMA-8B**, and **BART**. The performance of these models is evaluated based on the ROUGE metrics, with a focus on ROUGE F-measure, which reflects the balance between precision and recall.

## Model Performance Metrics

### Mistral-7B
- **ROUGE Precision**: 0.8925
- **ROUGE Recall**: 0.0455
- **ROUGE F-measure**: 0.0854

**Analysis**:
Mistral-7B exhibits very high precision, indicating that it generates highly accurate content. However, it has a low recall and F-measure, suggesting it misses relevant content and lacks overall balance in generating comprehensive subject lines.

### LLaMA-8B
- **ROUGE Precision**: 0.3250
- **ROUGE Recall**: 0.3166
- **ROUGE F-measure**: 0.3142

**Analysis**:
LLaMA-8B demonstrates the best overall balance among the models. It achieves relatively high precision, recall, and F-measure, making it effective at generating subject lines that are both accurate and comprehensive.

### BART
- **ROUGE Precision**: 0.1350
- **ROUGE Recall**: 0.0803
- **ROUGE F-measure**: 0.0919

**Analysis**:
BART has the lowest precision and recall of the models. Its F-measure is moderate, reflecting that while it generates some relevant content, it is less competitive compared to LLaMA-8B and Mistral-7B in terms of generating high-quality subject lines.

## Comparative Summary

- **LLaMA-8B**: With the highest F-measure (0.3142), LLaMA-8B is the best performer overall. It provides a balanced approach, excelling in both precision and recall, making it the top choice for generating subject lines that are accurate and relevant.

- **Mistral-7B**: Although it achieves high precision (0.8925), its low F-measure (0.0854) indicates that it may miss relevant content. It is suitable if precision is the primary focus but less so for balanced performance.

- **BART**: This model performs the least effectively in terms of precision and recall. Its moderate F-measure suggests that it is less competitive for generating subject lines compared to LLaMA-8B and Mistral-7B.

## Recommendation

- **LLaMA-8B** is recommended for generating subject lines if a balanced performance across precision, recall, and F-measure is desired.

- **Mistral-7B** is better suited if high precision is crucial, though it has a lower overall F-measure.

- **BART** is less favorable for subject line generation due to its lower performance metrics.

In summary, **LLaMA-8B** offers the best overall performance based on F-measure, providing a balanced approach between accuracy and relevance in subject line generation.


