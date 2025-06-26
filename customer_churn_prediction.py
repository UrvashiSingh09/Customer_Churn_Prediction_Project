# Customer Churn Prediction for a Telecom Company

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load your dataset
# Update the file path to where your CSV is located in Google Drive
df = pd.read_csv("/content/customer_ churn.csv")

# Step 3: Data Preprocessing
# Convert 'TotalCharges' to numeric (some entries might be non-numeric or empty)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)  # Drop rows with missing values

# Drop CustomerID as it's just an identifier
# Check if 'CustomerID' column exists before dropping
if 'CustomerID' in df.columns:
    df.drop(['CustomerID'], axis=1, inplace=True)


# Encode categorical columns using Label Encoding
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Step 4: Train-Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(accuracy * 100, 2), "%\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Feature Importance Visualization
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', color='lightblue')
plt.title("Top 10 Features Influencing Churn")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()

# Step 9: Business Recommendations
recommendations = [
    "1. Offer loyalty rewards to long-tenure customers.",
    "2. Improve support for users with Fiber optic service (higher churn).",
    "3. Encourage long-term contracts with incentives.",
    "4. Simplify and clarify billing processes."
]
print("\n💡 Recommendations to Reduce Churn:")
for rec in recommendations:
    print(rec)