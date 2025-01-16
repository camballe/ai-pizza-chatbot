import json
import pickle
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open("chatbot/intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
contexts = set()
ignore_letters = ["?", "!", ".", ","]

# Process intents and patterns
for intent in intents["intents"]:
    # Add context information
    if "requires_context" in intent:
        contexts.add(intent["requires_context"])
    if "context_set" in intent:
        contexts.add(intent["context_set"])
    
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Include context in documents
        context = intent.get("requires_context", "")
        documents.append((word_list, intent["tag"], context))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes and contexts
classes = sorted(set(classes))
contexts = sorted(contexts)

print(f"Unique words: {len(words)}")
print(f"Classes: {len(classes)}")
print(f"Contexts: {len(contexts)}")
print(f"Documents: {len(documents)}")

# Save words, classes, and contexts
pickle.dump(words, open("chatbot/words.pkl", "wb"))
pickle.dump(classes, open("chatbot/classes.pkl", "wb"))
pickle.dump(contexts, open("chatbot/contexts.pkl", "wb"))

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
    
    # Create context vector
    context_vector = [1 if document[2] == ctx else 0 for ctx in contexts]
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    # Append bag of words, context vector, and output
    training.append([bag, context_vector, output_row])

# Shuffle training data
random.shuffle(training)

# Convert to numpy arrays and split into inputs and output
train_x = np.array([item[0] for item in training])
train_context = np.array([item[1] for item in training])
train_y = np.array([item[2] for item in training])

print(f"Training data shapes:")
print(f"X (words): {train_x.shape}")
print(f"Context: {train_context.shape}")
print(f"Y (output): {train_y.shape}")

# Create model with context awareness
# Input layer for words
word_input = Input(shape=(len(train_x[0]),))
word_dense1 = Dense(128, activation="relu")(word_input)
word_dropout1 = Dropout(0.5)(word_dense1)

# Input layer for context
context_input = Input(shape=(len(contexts),))
context_dense1 = Dense(32, activation="relu")(context_input)
context_dropout1 = Dropout(0.5)(context_dense1)

# Combine word and context features
combined = Concatenate()([word_dropout1, context_dropout1])
combined_dense = Dense(64, activation="relu")(combined)
combined_dropout = Dropout(0.5)(combined_dense)

# Output layer
output = Dense(len(train_y[0]), activation="softmax")(combined_dropout)

# Create model
model = Model(inputs=[word_input, context_input], outputs=output)

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
hist = model.fit(
    [train_x, train_context],
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save model
model.save("chatbot/chatbot_model.h5")
print("Model training completed and saved to chatbot/chatbot_model.h5")