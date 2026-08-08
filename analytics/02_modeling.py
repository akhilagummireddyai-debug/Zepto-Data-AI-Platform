# ============================================================
# MODULE 2 - ANALYTICS PIPELINE
# MODELING
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.base import clone

import joblib


# -----------------------------
# 2. LOAD CLEANED DATA
# -----------------------------

# We use the same titanic.csv
# created by 01_eda.py

df = pd.read_csv("titanic.csv")

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# PART B - CLASSIFICATION
# ============================================================


# -----------------------------
# 3. DEFINE TARGET
# -----------------------------

# survived is our target column

target = "survived"

X = df.drop(columns=[target])

y = df[target]

print("\nTarget distribution:")
print(y.value_counts())


# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------

# Stratify keeps the same survived/not-survived
# proportion in train and test data.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -----------------------------
# 5. SELECT FEATURES
# -----------------------------

# We use these features for classification.

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# -----------------------------
# 6. PREPROCESSING
# -----------------------------

# Numeric:
# Missing values -> median
# Scaling -> StandardScaler

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical:
# Missing values -> most frequent
# Encoding -> One Hot Encoding

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


# Combine numeric + categorical preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 7. CREATE THREE CLASSIFICATION MODELS
# ============================================================


# -----------------------------
# Logistic Regression
# -----------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# -----------------------------
# Decision Tree
# -----------------------------

tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]
)


# -----------------------------
# Random Forest
# -----------------------------

forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# -----------------------------
# 8. TRAIN MODELS
# -----------------------------

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Training Decision Tree...")
tree_model.fit(X_train, y_train)

print("Training Random Forest...")
forest_model.fit(X_train, y_train)

print("All models trained successfully.")


# ============================================================
# 9. MODEL EVALUATION FUNCTION
# ============================================================


def evaluate_model(model, name):

    # Predict classes
    predictions = model.predict(X_test)

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n--------------------------------")
    print(name)
    print("--------------------------------")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC      :", round(auc, 4))

    # Confusion Matrix

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }


# -----------------------------
# 10. EVALUATE ALL THREE
# -----------------------------

results = []

results.append(
    evaluate_model(
        logistic_model,
        "Logistic Regression"
    )
)

results.append(
    evaluate_model(
        tree_model,
        "Decision Tree"
    )
)

results.append(
    evaluate_model(
        forest_model,
        "Random Forest"
    )
)


# -----------------------------
# 11. COMPARISON TABLE
# -----------------------------

classification_results = pd.DataFrame(results)

print("\n================================")
print("CLASSIFICATION COMPARISON")
print("================================")

print(
    classification_results.round(4)
)


# ============================================================
# 12. CONFUSION MATRICES
# ============================================================

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

for name, model in models.items():

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(name + " - Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        name.replace(" ", "_").lower()
        + "_confusion_matrix.png"
    )

    plt.close()


# ============================================================
# 13. ROC CURVES
# ============================================================

plt.figure(figsize=(7, 5))

for name, model in models.items():

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc_value:.3f}"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("roc_comparison.png")

plt.close()


# ============================================================
# 14. DECISION TREE VISUALIZATION
# ============================================================

# Get transformed feature names

trained_preprocessor = tree_model.named_steps[
    "preprocessor"
]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)


# Get trained decision tree

trained_tree = tree_model.named_steps[
    "classifier"
]


plt.figure(
    figsize=(20, 10)
)

plot_tree(
    trained_tree,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree")

plt.tight_layout()

plt.savefig(
    "decision_tree.png"
)

plt.close()


# ============================================================
# 15. IMBALANCE COMPARISON
# ============================================================

print("\n================================")
print("CLASS BALANCE")
print("================================")

print(
    y.value_counts()
)

print(
    y.value_counts(normalize=True) * 100
)


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test
)


# ------------------------------------------------------------
# Class Weight Balanced
# ------------------------------------------------------------

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)

balanced_model.fit(
    X_train,
    y_train
)

balanced_pred = balanced_model.predict(
    X_test
)


# ------------------------------------------------------------
# SMOTE
# ------------------------------------------------------------

# SMOTE is applied ONLY on training data.

try:

    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    smote_model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(
                max_iter=1000
            ))
        ]
    )

    smote_model.fit(
        X_train,
        y_train
    )

    smote_pred = smote_model.predict(
        X_test
    )

    smote_available = True

except ImportError:

    print(
        "\nSMOTE package not installed."
    )

    print(
        "Run: pip install imbalanced-learn"
    )

    smote_available = False


# -----------------------------
# Compare imbalance methods
# -----------------------------

imbalance_results = []


def add_imbalance_result(
    name,
    predictions
):

    imbalance_results.append({
        "Method": name,

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })


add_imbalance_result(
    "Baseline",
    baseline_pred
)

add_imbalance_result(
    "Class Weight Balanced",
    balanced_pred
)


if smote_available:

    add_imbalance_result(
        "SMOTE",
        smote_pred
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)

print("\n================================")
print("IMBALANCE COMPARISON")
print("================================")

print(
    imbalance_df.round(4)
)


# ============================================================
# 16. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n================================")
print("RANDOM FOREST GRID SEARCH")
print("================================")


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        )
    ]
)


param_grid = {

    "classifier__n_estimators": [
        50,
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)


best_rf = grid_search.best_estimator_


oob_score = (
    best_rf
    .named_steps["classifier"]
    .oob_score_
)


print(
    "\nOOB Score:",
    round(oob_score, 4)
)


# ============================================================
# 17. REGRESSION - PREDICT FARE
# ============================================================

print("\n================================")
print("REGRESSION")
print("================================")


# Target

regression_target = df["fare"]


# Features

regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]


X_reg = df[
    regression_features
]


y_reg = regression_target


# Split

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# Numeric features

reg_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]


# Categorical features

reg_categorical = [
    "sex",
    "embarked"
]


# Numeric pipeline

reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical pipeline

reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Preprocessor

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric
        ),

        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical
        )
    ]
)


# Regression pipeline

regression_model = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),

        (
            "regressor",
            LinearRegression()
        )
    ]
)


# Train

regression_model.fit(
    X_reg_train,
    y_reg_train
)


# Predict

reg_predictions = regression_model.predict(
    X_reg_test
)


# Metrics

mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        reg_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    reg_predictions
)


# Adjusted R2

n = len(y_reg_test)

p = (
    regression_model
    .named_steps["preprocessor"]
    .transform(
        X_reg_test
    ).shape[1]
)


adjusted_r2 = (
    1
    -
    (1 - r2)
    *
    (n - 1)
    /
    (n - p - 1)
)


print("\nMAE:", round(mae, 4))

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R2:",
    round(r2, 4)
)

print(
    "Adjusted R2:",
    round(adjusted_r2, 4)
)


# ============================================================
# 18. RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test
    -
    reg_predictions
)


plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    reg_predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    "regression_residuals.png"
)

plt.close()


# ============================================================
# 19. MODEL COMPARISON TABLE
# ============================================================

print("\n================================")
print("FINAL MODEL COMPARISON")
print("================================")


classification_table = classification_results[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]
].copy()


classification_table = (
    classification_table
    .round(4)
)


print("\nClassification Metrics:")

print(
    classification_table
)


regression_table = pd.DataFrame({

    "Model": ["Linear Regression"],

    "MAE": [mae],

    "RMSE": [rmse],

    "R2": [r2],

    "Adjusted_R2": [adjusted_r2]

})


print(
    "\nRegression Metrics:"
)

print(
    regression_table.round(4)
)


# ============================================================
# 20. SAVE BEST COMPLETE PIPELINE
# ============================================================

# We choose Random Forest as the final model.

final_pipeline = best_rf


joblib.dump(
    final_pipeline,
    "best_titanic_pipeline.joblib"
)


print(
    "\nBest complete pipeline saved successfully."
)


# ============================================================
# 21. RELOAD SAVED PIPELINE
# ============================================================

loaded_pipeline = joblib.load(
    "best_titanic_pipeline.joblib"
)


# Raw test data is directly passed.
# No manual preprocessing is required.

loaded_predictions = (
    loaded_pipeline
    .predict(X_test)
)


loaded_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)


print(
    "\nReloaded Pipeline Accuracy:",
    round(loaded_accuracy, 4)
)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n================================")
print("MODEL 2 COMPLETED SUCCESSFULLY")
print("================================")