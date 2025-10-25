import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# --- 1. Load Your Data ---
import dvc.api
with dvc.api.open('data/train.csv') as f:
    df = pd.read_csv(f)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df['comment_text']
y = df[label_cols]



# --- 2. Set Up MLflow Experiment (CORRECTED LOGIC) ---
experiment_name = "Toxicity_Classifier"
artifact_location = "gs://mlops-try/mlflow" 

# Try to get the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # Experiment doesn't exist, create it with the GCS artifact location
    print(f"Creating new experiment '{experiment_name}' with artifact location '{artifact_location}'")
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_location
    )
else:
    # Experiment already exists, just get its ID
    print(f"Using existing experiment: '{experiment_name}'")
    experiment_id = experiment.experiment_id

# Set the experiment as active *before* the run
mlflow.set_experiment(experiment_name=experiment_name)


# --- 3. Create Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 4. Start MLflow Run ---
with mlflow.start_run(experiment_id=experiment_id) as run:

    # --- 5. Define Parameters & Pipeline ---
    params = {
        "solver": "liblinear",
        "random_state": 42,
        "max_features": 5000,
        "stop_words": "english"
    }
    mlflow.log_params(params)

    vectorizer = TfidfVectorizer(
        stop_words=params["stop_words"],
        max_features=params["max_features"]
    )
    base_model = LogisticRegression(
        solver=params["solver"],
        random_state=params["random_state"]
    )
    classifier = MultiOutputClassifier(base_model)

    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', classifier)
    ])

    # --- 6. Train the Pipeline ---
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- 7. Evaluate and Log Metrics ---
    y_pred = pipeline.predict(X_test)
    report_dict = classification_report(y_test, y_pred, target_names=label_cols, zero_division=0, output_dict=True)
    
    f1_macro = report_dict['macro avg']['f1-score']
    precision_macro = report_dict['macro avg']['precision']
    recall_macro = report_dict['macro avg']['recall']
    
    mlflow.log_metric("f1_score_macro", f1_macro)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)

    print(f"\nF1-Score (Macro Avg): {f1_macro:.4f}")

    # --- 8. Log the Model (Pipeline) ---
    print("Logging model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model" 
    )
    
    run_id = run.info.run_id
    print(f"\nRun complete. Run ID: {run_id}")
    print(f"Model will be saved to: {artifact_location}/{run_id}/artifacts/model")

