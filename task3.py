import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
# Convert to a Pandas DataFrame for easier exploration
data = pd.DataFrame(
    data=iris.data, columns=iris.feature_names
)
data['target'] = iris.target

data['target_names'] = data['target'].apply(lambda x: iris.target_names[x])

# Step 2: Explore the dataset
print("Dataset Head:\n", data.head())
print("\nDescription of Features:\n", iris.DESCR)

# Step 3: Split the dataset
X = data[iris.feature_names]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Predict a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
predicted_class = model.predict(new_sample)
predicted_class_name = iris.target_names[predicted_class[0]]
print(f"\nNew sample prediction: {predicted_class_name}")
