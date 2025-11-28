import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# Load datasets
authentic_df = pd.read_csv('/content/LabeledAuthentic-7K.csv')  # Update path as needed
fake_df = pd.read_csv('/content/LabeledFake-1K.csv')  # Update path as needed

# Combine the datasets
df = pd.concat([authentic_df, fake_df], ignore_index=True)

# Preprocessing
df['text'] = df['headline'] + " " + df['content']
X = df['text'].values
y = df['label'].values

# Tokenization and padding
max_num_words = 50000
max_seq_len = 300
tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define the model with some improvements
embedding_dim = 100
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=max_num_words, output_dim=embedding_dim))

# CNN layer for feature extraction
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Batch Normalization layer
model.add(BatchNormalization())

# Bidirectional LSTM layer for sequential processing
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# Adding Dropout for regularization
model.add(Dropout(0.5))

# Bidirectional GRU layer to capture more dependencies
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())

# Dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a different optimizer
optimizer = AdamW(learning_rate=0.001)  # Experiment with different learning rates
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks for early stopping and reducing learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

# Train the model with increased epochs and validation split
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.15, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
