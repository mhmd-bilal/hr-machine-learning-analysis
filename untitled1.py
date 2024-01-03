# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('HR-Employee-Attrition.csv')

# Preprocessing
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Define features (X) and target variable (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualizations
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

# Random Forest Model
rf_classifier = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=3, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf = grid_search_rf.best_params_

# Train the model with the best hyperparameters
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params_rf)
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test set with Random Forest
y_pred_rf = best_rf_classifier.predict(X_test)

# Logistic Regression Model
logreg_classifier = LogisticRegression(random_state=42)

# Train the Logistic Regression model
logreg_classifier.fit(X_train, y_train)

# Make predictions on the test set with Logistic Regression
y_pred_logreg = logreg_classifier.predict(X_test)

# Evaluate the models
# Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf, target_names=['No', 'Yes'])

# Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
classification_rep_logreg = classification_report(y_test, y_pred_logreg, target_names=['No', 'Yes'])

# Print summary for Random Forest
print("Random Forest Model:")
print(f"Best Hyperparameters: {best_params_rf}")
print(f"Accuracy: {accuracy_rf:.2%}")
print("Classification Report:")
print(classification_rep_rf)

# Print summary for Logistic Regression
print("\nLogistic Regression Model:")
print(f"Accuracy: {accuracy_logreg:.2%}")
print("Classification Report:")
print(classification_rep_logreg)

# Visualize Accuracy
models = ['Random Forest', 'Logistic Regression']
accuracies = [accuracy_rf, accuracy_logreg]

axes[0, 0].bar(models, accuracies, color=['green', 'blue'])
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel('Accuracy')

# Count plot for Attrition
sns.countplot(x='Attrition', data=df, palette='viridis', ax=axes[0, 1], hue='Attrition', legend=False)
axes[0, 1].set_title('Attrition Count')
axes[0, 1].set_xlabel('Attrition')
axes[0, 1].set_ylabel('Count')

# Correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0, 2])
axes[0, 2].set_title('Correlation Heatmap')

# Pair Plot for Feature Distribution
features_for_pair_plot = ['Age', 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'TotalWorkingYears', 'Attrition']
sns.pairplot(df[features_for_pair_plot], hue='Attrition', palette='viridis')
axes[1, 0].set_title('Pair Plot of Selected Features')

# Confusion Matrix for Random Forest
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, ax=axes[1, 1])
axes[1, 1].set_title('Random Forest Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

# Confusion Matrix for Logistic Regression
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, ax=axes[1, 2])
axes[1, 2].set_title('Logistic Regression Confusion Matrix')
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('Actual')

plt.tight_layout()
plt.show()

