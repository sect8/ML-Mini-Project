import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['date'])  # Drop 'date' column
    
    # Encoding the target variable 'weather'
    label_encoder = LabelEncoder()
    df['weather'] = label_encoder.fit_transform(df['weather'])
    
    # Features and target separation
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y, label_encoder

def evaluate_model(X, y, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # 1. Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 3. Classification Report (Precision, Recall, F1-Score)
    cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(cr)

    # 4. ROC Curve and AUC Score
    y_test_bin = pd.get_dummies(y_test).values
    y_score = model.predict_proba(X_test_scaled)

    plt.figure(figsize=(10, 8))
    for i, weather in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {weather}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 5. Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.2f}")

if __name__ == "__main__":
    file_path = "D:/ML MINI PROJECT/Weather_Prediction/seattle-weather.csv"
    X, y, label_encoder = load_and_preprocess_data(file_path)
    evaluate_model(X, y, label_encoder)
