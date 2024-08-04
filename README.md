# Model Performance Comparison

## Background

This project involves training three variants of large language models (LLMs) to predict email subject lines given the email body. The models used are:

- **Mistral-7B**
- **LLaMA-3B**
- **GPT-2**

The training was performed using the [AESLC](https://github.com/ryanzhumich/AESLC) dataset. This dataset provides a corpus for email subject line prediction, enabling us to fine-tune pre-trained models to improve their performance on this specific task.

## Training Details

The models were trained and fine-tuned using the provided Python code. The training process involved:

1. **Data Preprocessing**: The AESLC dataset was prepared by extracting the email bodies and their corresponding subject lines.
2. **Fine-Tuning**: Each pre-trained model was fine-tuned on the AESLC dataset using custom training scripts.
3. **Evaluation**: The performance of the models was evaluated using the ROUGE metric, focusing on precision, recall, and F-measure.

## Performance Comparison

The performance of the models was evaluated in two scenarios:
1. **Pre-trained Models with Fine-Tuning**: The pre-trained models were fine-tuned using the AESLC corpus.
2. **Models Trained Using AESLC Corpus**: The models were trained from scratch using the AESLC corpus.

The results are summarized in the table below:

| Model      | Training Method                  | ROUGE Precision | ROUGE Recall | ROUGE F-measure |
|------------|----------------------------------|-----------------|--------------|-----------------|
| Mistral-7B | Pre-trained with Fine-Tuning     | 0.8837          | 0.0478       | 0.0897          |
| Mistral-7B | Trained Using AESLC Corpus       | 0.8925          | 0.0455       | 0.0854          |
| LLaMA-3B   | Pre-trained with Fine-Tuning     | 0.3118          | 0.3915       | 0.3172          |
| LLaMA-3B   | Trained Using AESLC Corpus       | 0.19            | 0.0169       | 0.0292          |
| GPT-2      | Pre-trained with Fine-Tuning     | XX.XX           | XX.XX        | XX.XX           |
| GPT-2      | Trained Using AESLC Corpus       | XX.XX           | XX.XX        | XX.XX           |

*Note: Replace "XX.XX" with actual metric values obtained from your evaluations for GPT-2.*

## Python Training Code

The Python code used for training and fine-tuning the models is available in this repository. The main scripts include:

- `preprocess_data.py`: Script for preprocessing the AESLC dataset.
- `train_model.py`: Script for training the models.
- `evaluate_model.py`: Script for evaluating the model performance.

## Conclusion

This project demonstrates the effectiveness of fine-tuning pre-trained language models using a specialized corpus like AESLC for the task of email subject line prediction. The comparison of ROUGE metrics highlights the improvements achieved through fine-tuning, providing insights into the performance of different model variants.

For more details, please refer to the individual scripts and their documentation within this repository.
