# Sentiment Analysis Using LLM on Amazon Reviews

This project performs binary sentiment classification on Amazon product reviews using the RoBERTa model. It tackles class imbalance through resampling and data augmentation, achieving high overall accuracy and improved performance on the minority class.

- Fine-tune RoBERTa (`roberta-base`) on Amazon product reviews.
- Apply data augmentation and sampling techniques to address class imbalance.
- Evaluate the model with focus on minority class performance.


### Class Imbalance 

- **Random Oversampling** of minority class
- **Back-Translation** for data augmentation (English → German → English)
- **Weighted Cross-Entropy Loss** during training

### Model

- Pre-trained `roberta-base` from Hugging Face Transformers
- Fine-tuned using PyTorch and Hugging Face `Trainer` API

### Results

| Metric                  | Value     |
|-------------------------|-----------|
| **Accuracy**            | 96%       |
| **F1 Score (Minority)** | 0.60      |
| **F1 Score (Majority)** | 0.98      |

The use of class balancing techniques significantly improved the model’s ability to correctly classify underrepresented sentiment.

### Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Google Colab (for training and GPU support)

### Key Insights

- Class balancing is critical when working with real-world, imbalanced datasets.
- Back-translation increased data diversity and improved generalization.
- RoBERTa showed excellent performance after minimal fine-tuning.




