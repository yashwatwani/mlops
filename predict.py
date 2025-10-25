import mlflow
import sys
import pandas as pd

# --- 1. Configuration ---
# PASTE YOUR RUN ID HERE
RUN_ID = "eae1601f40eb4fdcb026f672b9fb734a" 

# This must match the database file you're using
TRACKING_URI = "sqlite:///mlflow.db" 

# This is the artifact path we set in train_mlflow.py
ARTIFACT_PATH = "model"

# These are the labels your model predicts
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- 2. Check for Input ---
if len(sys.argv) < 2:
    print("\n--- Error: No comment provided. ---")
    print(f"Usage: python {sys.argv[0]} \"Your comment to test\"")
    print("Example: python predict.py \"This is a horrible comment!\"")
    sys.exit(1)

comment_text = sys.argv[1]

# --- 3. Load Model ---
try:
    # Set the tracking URI to find the local database
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Construct the full model URI
    model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
    print(f"Loading model from: {model_uri}")

    # Load the model (which is the full sklearn pipeline)
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")

except Exception as e:
    print(f"--- Error loading model ---")
    print(f"Could not find Run ID: {RUN_ID}")
    print("Please make sure your RUN_ID is correct and 'mlflow.db' exists.")
    print(f"Full error: {e}")
    sys.exit(1)


# --- 4. Make Prediction ---
# We predict on a list containing the single comment
# The model.predict() expects a list or 1D array of text
prediction_array = model.predict([comment_text])

# The output is a 2D array (e.g., [[1, 0, 1, 0, 0, 0]]), so get the first item
prediction = prediction_array[0]

# --- 5. Format and Print Results ---
print("\n" + "="*30)
print(f"Comment: \"{comment_text}\"")
print("-"*30)

predicted_labels = []
for label, value in zip(LABELS, prediction):
    if value == 1:
        predicted_labels.append(label)

if predicted_labels:
    print(f"Prediction: {', '.join(predicted_labels)}")
else:
    print("Prediction: (Clean)")
print("="*30)