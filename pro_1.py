# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import numpy as np
# numpy -> used for numerical operations and array handling

from tensorflow.keras.datasets import imdb
# imdb -> built-in Keras dataset for movie review sentiment analysis


# =====================
# LOAD IMDB DATASET
# =====================

max_features = 1000
# Maximum number of most frequent words to keep in vocabulary

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# Load IMDB dataset and restrict vocabulary size

print(f"Training data shape: {len(X_train)}, Training labels shape: {len(y_train)}")
print(f"Testing data shape: {len(X_test)}, Testing labels shape: {len(y_test)}")


# =====================
# COMBINE TRAIN & TEST DATA
# =====================

X = np.concatenate((X_train, X_test), axis=0)
# Combine all reviews into a single array

y = np.concatenate((y_train, y_test), axis=0)
# Combine all labels into a single array


# =====================
# MANUAL TRAIN-TEST SPLIT
# =====================

split_index = int(0.8 * len(X))
# Calculate index for 80% training data

X_train = X[:split_index]
y_train = y[:split_index]
# Assign first 80% as training set

X_test = X[split_index:]
y_test = y[split_index:]
# Assign remaining 20% as test set

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================
# PAD SEQUENCES
# =====================

from tensorflow.keras.preprocessing import sequence
# sequence -> used to pad variable-length sequences

max_len = 200
# Maximum length of each review sequence

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
# Pad training reviews to fixed length

X_test = sequence.pad_sequences(X_test, maxlen=max_len)
# Pad testing reviews to fixed length

print(X_train.shape)
# Verify padded input shape


# =====================
# BUILD RNN MODEL
# =====================

from tensorflow.keras.models import Sequential
# Sequential -> linear stack of layers

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
# Embedding -> converts word indices to dense vectors
# SimpleRNN -> recurrent layer for sequence learning
# Dense -> fully connected output layer

model = Sequential()
# Initialize Sequential model

model.add(Embedding(input_dim=max_features, output_dim=128))
# Embedding layer to learn word representations

model.add(SimpleRNN(128, activation='tanh'))
# Simple RNN layer to capture sequence dependencies

model.add(Dense(1, activation='sigmoid'))
# Output layer for binary classification


# =====================
# COMPILE MODEL
# =====================

model.compile(
    optimizer='adam',
    # Adam optimizer for efficient gradient descent

    loss='binary_crossentropy',
    # Binary Crossentropy for binary sentiment classification

    metrics=['accuracy']
    # Track accuracy during training
)

model.summary()
# Display model architecture


# =====================
# EARLY STOPPING CALLBACK
# =====================

from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping -> stops training when validation loss stops improving

earlystopping = EarlyStopping(
    monitor='val_loss',
    # Monitor validation loss

    patience=5,
    # Stop if no improvement for 5 epochs

    restore_best_weights=True
    # Restore best weights after stopping
)


# =====================
# TRAIN MODEL
# =====================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    # Maximum number of epochs

    batch_size=32,
    # Number of samples per batch

    validation_split=0.2,
    # Use 20% of training data for validation

    callbacks=[earlystopping]
    # Apply early stopping
)


# =====================
# VISUALIZE TRAINING RESULTS
# =====================

import matplotlib.pyplot as plt
# matplotlib -> used for plotting graphs

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Extract training history values

epochs = range(1, len(accuracy) + 1)
# Create epoch range


# ---- Accuracy Plot ----
plt.figure(figsize=(7,4))
plt.plot(epochs, accuracy, marker='o', label='Training Accuracy')
plt.plot(epochs, val_accuracy, marker='s', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# ---- Loss Plot ----
plt.figure(figsize=(7,4))
plt.plot(epochs, loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# =====================
# SAVE AND LOAD MODEL
# =====================

model.save("simple_rnn_imdb.h5")
# Save trained model to disk

from tensorflow.keras.models import load_model
# load_model -> reload saved model

model = load_model("simple_rnn_imdb.h5")
# Load the saved model


# =====================
# EVALUATE MODEL
# =====================

test_loss, test_accuracy = model.evaluate(X_test, y_test)
# Evaluate model on test data

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# =====================
# SINGLE REVIEW PREDICTION
# =====================

sample_review = X_test[1]
# Select one review from test set

prediction = model.predict(sample_review.reshape(1, -1))
# Predict sentiment probability

sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
# Convert probability into sentiment label

print("Predicted Sentiment:", sentiment)
print("Actual Label:", "Positive" if y_test[1] == 1 else "Negative")