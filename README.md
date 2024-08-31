# Model Performance Comparison

## Background

This project involves training three variants of large language models (LLMs) to predict email subject lines given the email body. The models used are:

- **Mistral-7B**
- **LLaMA-8B**
- **BART**
- **t5-small**

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
| BART       | Pre-trained with Fine-tuning     | 0.0246          | 0.5319       | 0.0456          |
| BART       | Trained with AESLC Corpos        | 0.3039          | 0.3788       | 0.3079          |
| t5-small   | Pre-trained                      | 0.6130          | 0.6977       | 0.1225          |
| t5-small   | Trained Using AESLC Corpus       | 0.5296          | 0.6229       | 0.5456          |


## Python Training Code

The Python code used for training and fine-tuning the models is available in this repository. The main scripts include:

- `preprocess_data.py`: Script for preprocessing the AESLC dataset.
- `train_model.py`: Script for training the models.
- `evaluate_model.py`: Script for evaluating the model performance.


# Model Performance Analysis for Subject Line Prediction

This document provides a comparative analysis of four models: **Mistral-7B**, **LLaMA-8B**, **BART**, and **T5-Small** for the task of predicting email subject lines. The models are evaluated based on ROUGE metrics: Precision, Recall, and F-measure.

## Model Performance Metrics

### Mistral-7B
- **ROUGE Precision**: 0.8925
- **ROUGE Recall**: 0.0455
- **ROUGE F-measure**: 0.0854

**Analysis**:  
Mistral-7B achieves very high precision, suggesting it generates highly accurate content. However, its low recall and F-measure indicate that it misses a significant amount of relevant content, resulting in a lack of balance for generating comprehensive subject lines.

### LLaMA-8B
- **ROUGE Precision**: 0.3325
- **ROUGE Recall**: 0.3166
- **ROUGE F-measure**: 0.3142

**Analysis**:  
LLaMA-8B offers the most balanced performance among the models, with relatively high precision, recall, and F-measure. It effectively generates subject lines that are both accurate and comprehensive.

### BART
- **ROUGE Precision**: 0.1350
- **ROUGE Recall**: 0.0803
- **ROUGE F-measure**: 0.0919

**Analysis**:  
BART demonstrates the lowest precision and recall among the models. Its F-measure is moderate, reflecting that it generates some relevant content but is generally less competitive compared to other models like LLaMA-8B and Mistral-7B.

### T5-Small
- **ROUGE Precision**: 0.5296
- **ROUGE Recall**: 0.6229
- **ROUGE F-measure**: 0.5456

**Analysis**:  
T5-Small shows a strong balance between precision (0.5296) and recall (0.6229), resulting in a relatively high F-measure (0.5456). This suggests that T5-Small is effective at generating subject lines that are both accurate and comprehensive, outperforming BART in all metrics and providing a balanced alternative to LLaMA-8B.

## Comparative Summary

- **T5-Small**:  
  T5-Small emerges as a strong performer with a higher F-measure (0.5456) than all other models, suggesting it effectively balances precision and recall. This model is highly effective for generating subject lines that are both accurate and comprehensive.

- **LLaMA-8B**:  
  Although its F-measure (0.3142) is lower than T5-Small, LLaMA-8B still offers a balanced performance across precision and recall, making it a good option for subject line generation when more balanced metrics are required.

- **Mistral-7B**:  
  With very high precision (0.8925) but a low F-measure (0.0854), Mistral-7B is suitable if precision is the most critical factor, although it lacks in recall and overall balance.

- **BART**:  
  With the lowest performance metrics across precision, recall, and F-measure, BART is less favorable for generating subject lines compared to T5-Small, LLaMA-8B, and Mistral-7B.

## Recommendation

1. **T5-Small** is the top recommendation for generating subject lines due to its strong balance of precision and recall, resulting in the highest F-measure (0.5456). It is ideal for users seeking an effective model for producing accurate and comprehensive subject lines.

2. **LLaMA-8B** is a solid choice when a balanced approach is desired, though it is slightly less effective than T5-Small in terms of F-measure.

3. **Mistral-7B** is recommended only if the focus is on high precision, with the understanding that it will likely miss some relevant content.

4. **BART** is the least favorable option given its lower performance metrics.

## Conclusion

Based on the ROUGE metrics, **T5-Small** is the best model for generating subject lines, providing a strong balance of precision
