import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# Dataset 1: Personal Loan Acceptance Prediction
# ============================================================

def load_personal_loan_dataset():
    """
    Personal Loan dataset ko direct CSV file se load karna.
    Aap ke folder mein file ka naam hai:
    Bank_Personal_Loan_Modelling.csv
    """

    csv_path = os.path.join(RAW_DATA_DIR, "Bank_Personal_Loan_Modelling.csv")

    if not os.path.exists(csv_path):
        available_files = os.listdir(RAW_DATA_DIR)

        raise FileNotFoundError(
            "Bank_Personal_Loan_Modelling.csv file data/raw folder mein nahi mili.\n"
            f"Available files: {available_files}"
        )

    df = pd.read_csv(csv_path)

    print("\nPersonal Loan Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


def clean_personal_loan_data(df):
    """
    Personal Loan dataset ki cleaning.
    ID aur ZIP Code prediction ke liye useful nahi hain,
    is liye unko remove kar rahe hain.
    """

    df = df.copy()

    # Unnecessary columns remove karna
    columns_to_drop = ["ID", "ZIP Code"]

    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Negative experience ko 0 karna
    if "Experience" in df.columns:
        df["Experience"] = df["Experience"].apply(lambda x: 0 if x < 0 else x)

    # Missing values remove karna
    df.dropna(inplace=True)

    print("\nPersonal Loan Dataset cleaned successfully.")
    print("Cleaned Shape:", df.shape)

    return df


def train_personal_loan_model(df):
    """
    Logistic Regression model train karna.
    Target column: Personal Loan
    """

    target_column = "Personal Loan"

    if target_column not in df.columns:
        raise ValueError(f"{target_column} column dataset mein nahi mila.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n===================================================")
    print("Personal Loan Acceptance Model Results")
    print("===================================================")
    print("Algorithm: Logistic Regression")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Model save karna
    model_path = os.path.join(MODELS_DIR, "personal_loan_model.pkl")
    joblib.dump(model, model_path)

    # Columns save karna dashboard ke liye
    columns_path = os.path.join(MODELS_DIR, "personal_loan_columns.pkl")
    joblib.dump(list(X.columns), columns_path)

    # Cleaned data save karna dashboard aur EDA ke liye
    processed_path = os.path.join(PROCESSED_DATA_DIR, "personal_loan_cleaned.csv")
    df.to_csv(processed_path, index=False)

    print("\nPersonal Loan model saved successfully.")

    return accuracy


# ============================================================
# Dataset 2: Loan Approval Prediction
# ============================================================

def load_loan_approval_dataset():
    """
    Loan Approval dataset ko CSV file se load karna.
    Aap ke folder mein file ka naam hai:
    Loan Approval Prediction.csv
    """

    csv_path = os.path.join(RAW_DATA_DIR, "Loan Approval Prediction.csv")

    if not os.path.exists(csv_path):
        available_files = os.listdir(RAW_DATA_DIR)

        raise FileNotFoundError(
            "Loan Approval Prediction.csv file data/raw folder mein nahi mili.\n"
            f"Available files: {available_files}"
        )

    df = pd.read_csv(csv_path)

    print("\nLoan Approval Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


def clean_loan_approval_data(df):
    """
    Loan Approval dataset ki cleaning.
    Ye dataset already encoded hai, is liye simple missing values handle kar rahe hain.
    """

    df = df.copy()

    # Missing values remove karna
    df.dropna(inplace=True)

    print("\nLoan Approval Dataset cleaned successfully.")
    print("Cleaned Shape:", df.shape)

    return df


def train_loan_approval_model(df):
    """
    Decision Tree Classifier train karna.
    Target column: loan_status
    """

    target_column = "loan_status"

    if target_column not in df.columns:
        raise ValueError(f"{target_column} column dataset mein nahi mila.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n===================================================")
    print("Loan Approval Prediction Model Results")
    print("===================================================")
    print("Algorithm: Decision Tree Classifier")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Model save karna
    model_path = os.path.join(MODELS_DIR, "loan_approval_model.pkl")
    joblib.dump(model, model_path)

    # Columns save karna dashboard ke liye
    columns_path = os.path.join(MODELS_DIR, "loan_approval_columns.pkl")
    joblib.dump(list(X.columns), columns_path)

    # Cleaned data save karna dashboard aur EDA ke liye
    processed_path = os.path.join(PROCESSED_DATA_DIR, "loan_approval_cleaned.csv")
    df.to_csv(processed_path, index=False)

    print("\nLoan Approval model saved successfully.")

    return accuracy


# ============================================================
# Main Function
# ============================================================

def main():
    print("Project training start ho rahi hai...")

    # -------------------------------
    # Personal Loan Acceptance Model
    # -------------------------------

    personal_df = load_personal_loan_dataset()
    personal_df = clean_personal_loan_data(personal_df)
    personal_accuracy = train_personal_loan_model(personal_df)

    # -------------------------------
    # Loan Approval Prediction Model
    # -------------------------------

    approval_df = load_loan_approval_dataset()
    approval_df = clean_loan_approval_data(approval_df)
    approval_accuracy = train_loan_approval_model(approval_df)

    # -------------------------------
    # Final Results Save
    # -------------------------------

    results = pd.DataFrame({
        "Model": [
            "Personal Loan Acceptance Prediction",
            "Loan Approval Prediction"
        ],
        "Algorithm": [
            "Logistic Regression",
            "Decision Tree Classifier"
        ],
        "Accuracy": [
            round(personal_accuracy * 100, 2),
            round(approval_accuracy * 100, 2)
        ]
    })

    results_path = os.path.join(REPORTS_DIR, "model_results.csv")
    results.to_csv(results_path, index=False)

    print("\n===================================================")
    print("Training complete ho gayi.")
    print("===================================================")
    print("Models models folder mein save ho gaye hain.")
    print("Processed data data/processed folder mein save ho gaya hai.")
    print("Results reports/model_results.csv mein save ho gaye hain.")


if __name__ == "__main__":
    main()