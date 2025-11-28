#A Robust Hybrid Deep Learning Framework for Bangla Fake News Detection

This project implements a CNN–LSTM hybrid deep learning model to identify fake vs. authentic news written in the Bengali language.
Bangla, with its complex morphology and low-resource ecosystem, presents unique challenges such as sarcasm, contextual ambiguity, idioms, and linguistic variations.
Our proposed model overcomes these challenges using a carefully designed preprocessing pipeline and a powerful deep neural architecture.

📌 Project Highlights

Datasets Used:

LabeledAuthentic-7K (7,000 real news articles)

LabeledFake-1K (1,000 fake news articles)

Preprocessing Steps:

Merge headline + content → text

Tokenizer with vocab size = 10,000

Padding length = 150

Train–test split = 80/20

Model Highlights:

Embedding layer

1D Convolution + MaxPooling

Bidirectional LSTM

GlobalAveragePooling

Dense layers with dropout

Adam optimizer + Binary Cross-Entropy

📊 Performance Summary

Accuracy: 96%

Fake News: Precision 0.95 | Recall 0.75

Authentic News: Precision 0.96 | Recall 0.99

Strong overall F1-score = 0.96 (weighted)

🚀 Future Scope

Data augmentation & oversampling

Transformer-based models (Bangla-BERT, DistilBERT)

Multiclass fake news categorization

Real-world deployment & web API integration
