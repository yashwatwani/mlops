import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle

# --- 1. Load Your Data ---
# I've manually recreated the data from your image for this example.
# In your real project, you would load your full CSV file like this:
# df = pd.read_csv('your_file.csv')


df = pd.read_csv('data/train.csv')

# --- 2. Define Features (X) and Labels (y) ---
# --- 2. Define Features (X) and Labels (y) ---
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df['comment_text']
y = df[label_cols]

# --- 3. Create Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 4. Vectorize the Text Data (TF-IDF) ---
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- 5. Build and Train the Model ---
base_model = LogisticRegression(solver='liblinear', random_state=42)
model = MultiOutputClassifier(base_model)

print("Training model...")
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 5a. Save the Model and Vectorizer to Pickle Files ---
# We must save BOTH the model and the vectorizer.
# The vectorizer is needed to transform new text in the *exact same way*
# as the text the model was trained on.

model_filename = 'toxicity_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

print(f"\nSaving model to {model_filename}...")
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"Saving vectorizer to {vectorizer_filename}...")
with open(vectorizer_filename, 'wb') as f:
    pickle.dump(vectorizer, f)
    
print("Files saved.")

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred, target_names=label_cols, zero_division=0)
print(report)

# --- 7. Test on New Comments (Original) ---
print("\n--- Test on New Data (using original model) ---")
new_comments = [
    "This is a wonderful and positive comment.",
    "You are a horrible person and I hate you."
]

new_comments_tfidf = vectorizer.transform(new_comments)
new_predictions = model.predict(new_comments_tfidf)

for comment, prediction in zip(new_comments, new_predictions):
    print(f"\nComment: '{comment}'")
    predicted_labels = [label for label, value in zip(label_cols, prediction) if value == 1]
    if predicted_labels:
        print(f"  Predicted as: {', '.join(predicted_labels)}")
    else:
        print("  Predicted as: (Clean)")

# --- 8. Load Model and Test (New Section) ---
# This simulates loading your saved model in a different script.
print("\n--- Test on New Data (using LOADED model) ---")

# Load the saved model and vectorizer
with open(model_filename, 'rb') as f:
    loaded_model = pickle.load(f)
    
with open(vectorizer_filename, 'rb') as f:
    loaded_vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully.")

# Prepare new comments with the LOADED vectorizer
loaded_comments_tfidf = loaded_vectorizer.transform(new_comments)

# Make predictions with the LOADED model
loaded_predictions = loaded_model.predict(loaded_comments_tfidf)

# Display results (should be identical to Step 7)
for comment, prediction in zip(new_comments, loaded_predictions):
    print(f"\nComment: '{comment}'")
    predicted_labels = [label for label, value in zip(label_cols, prediction) if value == 1]
    if predicted_labels:
        print(f"  Predicted as: {', '.join(predicted_labels)}")
    else:
        print("  Predicted as: (Clean)")