import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Paths
data_path = r"C:\Users\danny\OneDrive - Johns Hopkins\2025 Summer Semester\Class Project\Raw Data\headache_data.csv"
model_dir = r"C:\Users\danny\OneDrive - Johns Hopkins\2025 Summer Semester\Class Project\Training Data\Models"
plot_dir = r"C:\Users\danny\OneDrive - Johns Hopkins\2025 Summer Semester\Class Project\Training Data\Plots"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Drop non-numeric columns
X = df.drop(columns=["Timestamp", "Headache_Label"])
y = df["Headache_Label"]

# 80/20 Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
}

# Evaluation storage
results = {}

# K-Fold setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="accuracy")
    mean_cv_score = np.mean(cv_scores)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Headache", "Headache"])
    plt.yticks(tick_marks, ["No Headache", "Headache"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(plot_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, f"{name}_roc_curve.png"))
        plt.close()
    else:
        roc_auc = None

    # Store results
    results[name] = {
        "model": model,
        "cv_score": mean_cv_score,
        "accuracy": model.score(X_test_scaled, y_test),
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

# Select best model by cross-validation score
best_model_name = max(results, key=lambda k: results[k]["cv_score"])
best_model = results[best_model_name]["model"]

# Save best model as dictionary with scaler and label encoder
model_bundle = {
    "model": best_model,
    "scaler": scaler,
    "label_encoder": None  # No label encoder used for binary classification
}
with open(os.path.join(model_dir, f"{best_model_name}_bundle.pkl"), "wb") as f:
    pickle.dump(model_bundle, f)

# Print summary
print(f"Best model: {best_model_name}")
print(f"Cross-validation accuracy: {results[best_model_name]['cv_score']:.2f}")
print(f"Test accuracy: {results[best_model_name]['accuracy']:.2f}")