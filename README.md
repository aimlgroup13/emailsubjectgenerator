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


**Conclusion**: 
The Mistral-7B model fine-tuned on the pre-trained model slightly outperforms the one trained using the AESLC corpus. The differences in ROUGE metrics are minimal, but the fine-tuned model shows a better balance overall.

### LLaMA-8B

- **Pre-trained (without fine-tuning)**:
  - ROUGE Precision: 0.2645
  - ROUGE Recall: 0.2120
  - ROUGE F-measure: 0.2133

- **Trained Using AESLC Corpus**:
  - ROUGE Precision: 0.3250
  - ROUGE Recall: 0.3166
  - ROUGE F-measure: 0.3142

## Conclusion

- **Mistral-7B**: Exhibits a much higher ROUGE Precision (0.8925), indicating that it generates more accurate content with fewer errors. However, it shows significantly lower ROUGE Recall (0.0455) and F-measure (0.0854), suggesting it may miss relevant content or lack overall balance.

- **LLaMA-8B**: Demonstrates more balanced performance across all three metrics, with ROUGE Precision (0.3250), Recall (0.3166), and F-measure (0.3142) being relatively close. This balance indicates that LLaMA-8B is more consistent in generating comprehensive and accurate outputs.

### Which Performed Better?

- **Mistral-7B**: If the priority is **precision** (i.e., generating accurate content with fewer errors), Mistral-7B is the better performer.
  
- **LLaMA-8B**: If a more **balanced** performance across precision, recall, and F-measure is desired, LLaMA-8B outperforms Mistral-7B when trained with the AESLC dataset.

### Recommendation for Subject Line Generation

In the context of generating subject lines, **Mistral-7B** is recommended if you prioritize accuracy and precision, ensuring that the generated subject lines are concise and on point. However, if you require a model that is more consistent in capturing relevant content comprehensively, **LLaMA-8B** would be the better choice.

In summary, Mistral-7B excels in precision, while LLaMA-8B offers a better overall balance in performance metrics.

