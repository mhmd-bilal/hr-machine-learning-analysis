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

Func = open("output.html","w")  
Func.write(f"Random Forest Model:<br>Best Hyperparameters: {best_params_rf}<br>Accuracy: {accuracy_rf:.2%}<br>Classification Report: {classification_rep_rf}")
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
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='viridis', annot_kws={"size": 16}, ax=axes[1, 1])
axes[1, 1].set_title('Random Forest Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

# Confusion Matrix for Logistic Regression
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='viridis', annot_kws={"size": 16}, ax=axes[1, 2])
axes[1, 2].set_title('Logistic Regression Confusion Matrix')
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('Actual')


import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Random Forest Accuracy Bar Chart
accuracy_chart = px.bar(
    x=models,
    y=accuracies,
    color=models,
    text=[f'{acc:.2%}' for acc in accuracies],
    labels={'x': 'Models', 'y': 'Accuracy'},
    title='Model Accuracy Comparison',
    color_discrete_map={'Random Forest': '#42047E', 'Logistic Regression': '#07F49E'}  # Updated color codes
)

# Count plot for Attrition
count_plot = px.bar(
    df,
    x='Attrition',
    color='Attrition',
    title='Attrition Count',
    labels={'x': 'Attrition', 'y': 'Count'},
    color_continuous_scale=["#42047E", "#07F49E"]  # Updated color codes
)

heatmap = px.imshow(
    correlation_matrix.values,
    labels=dict(color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    color_continuous_scale=["#42047E", "#07F49E"]  # Updated color codes
)

# Pair Plot for Feature Distribution
pair_plot = px.scatter_matrix(
    df[features_for_pair_plot],
    dimensions=features_for_pair_plot,
    color='Attrition',
    title='Pair Plot of Selected Features',
    color_continuous_scale=["#42047E", "#07F49E"]  # Updated color codes
)

# Confusion Matrix for Random Forest
conf_matrix_rf_chart = px.imshow(
    conf_matrix_rf,
    labels=dict(color="Count"),
    x=['No', 'Yes'],
    y=['No', 'Yes'],
    color_continuous_scale=["#42047E", "#07F49E"]  # Updated color codes
)

# Confusion Matrix for Logistic Regression
conf_matrix_logreg_chart = px.imshow(
    conf_matrix_logreg,
    labels=dict(color="Count"),
    x=['No', 'Yes'],
    y=['No', 'Yes'],
    color_continuous_scale=["#42047E", "#07F49E"]  # Updated color codes
)

# Box Plot for MonthlyIncome distribution based on Attrition
box_plot = px.box(
    df,
    x='Attrition',
    y='MonthlyIncome',
    color='Attrition',
    title='MonthlyIncome Distribution based on Attrition',
    labels={'Attrition': 'Attrition', 'MonthlyIncome': 'Monthly Income'},
    color_discrete_map={'No': '#42047E', 'Yes': '#07F49E'}  # Updated color codes
)

# Save individual plots to HTML files
accuracy_chart.write_html('accuracy_chart.html')
count_plot.write_html('count_plot.html')
heatmap.write_html('heatmap.html')
pair_plot.write_html('pair_plot.html')
conf_matrix_rf_chart.write_html('conf_matrix_rf_chart.html')
conf_matrix_logreg_chart.write_html('conf_matrix_logreg_chart.html')
box_plot.write_html('box_plot.html')
