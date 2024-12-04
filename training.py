import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Process intents and patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and lower each word and remove duplicates
words = [
    lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters
]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

print(f"Unique words: {len(words)}")
print(f"Classes: {len(classes)}")
print(f"Documents: {len(documents)}")

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]

    # Create bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Convert to numpy array and split into X and y
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print(f"Training data shape - X: {train_x.shape}, Y: {train_y.shape}")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save("chatbot_model.h5", hist)
print("Model training completed and saved to chatbot_model.h5")
