import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Load the dataset
data = pd.read_csv('D:/ML MINI PROJECT/Weather_Prediction/seattle-weather.csv')

# Features: Precipitation, Temp_max, Temp_min, Wind
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
X = data[features].values

# Target: Binary classification of weather (Rain = 1, No Rain = 0)
# We will classify if the weather is "rain" or not
data['is_rain'] = data['weather'].apply(lambda x: 1 if x == 'rain' else 0)
y = data['is_rain'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Function to plot decision boundary (only works for 2D feature space, we'll skip this for higher dimensions)
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary of Logistic Regression')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Predict probabilities and plot probability curve
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Plot the distribution of predicted probabilities
plt.figure(figsize=(8,6))
sns.histplot(y_prob, bins=10, kde=True)
plt.title('Probability Distribution of Logistic Regression Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()
