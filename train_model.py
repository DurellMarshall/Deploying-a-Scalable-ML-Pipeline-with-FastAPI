from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / "census.csv"
print(data_path)
data = pd.read_csv(data_path)

train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["salary"],
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = train_model(X_train, y_train)

# save the model and the encoder
model_path = project_path / "model" / "model.pkl"
save_model(model, model_path)
encoder_path = project_path / "model" / "encoder.pkl"
save_model(encoder, encoder_path)
lb_path = project_path / "model" / "lb.pkl"
save_model(lb, lb_path)

# load the model
model = load_model(model_path)

preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

slice_output_path = project_path / "slice_output.txt"
with open(slice_output_path, "w", encoding="utf-8") as f:
    print(f"Overall precision: {p:.4f}", file=f)
    print(f"Overall recall: {r:.4f}", file=f)
    print(f"Overall F1: {fb:.4f}", file=f)
    print("", file=f)

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model,
        )
        with open(slice_output_path, "a", encoding="utf-8") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
