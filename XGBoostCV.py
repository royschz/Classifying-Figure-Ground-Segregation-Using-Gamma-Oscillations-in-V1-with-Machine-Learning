# Import necessary libraries
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.metrics import f1_score, matthews_corrcoef, auc
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')

#%% XGBoost performing. 
# Data frame with the feature selection. 
X = data[['ContrastHeterogeneity', 'GridCoarseness']]

# Define the target variable “y”, that contains the binary values.  
y = data['Correct']

# Split the data set into training (80%) and test (20%), stratify the “y” data set to balance the data from the “y_train”. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameters with Cross-Validation. 
# Establish the XGBoost Classifier. 
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define the parameters that we want to apply. 
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set the GridSearchCV to perform a search over the hyperparameters, using a 5 fold for the cross-validation. 
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the train data set.  
grid_search.fit(X_train, y_train)

# Find the best hyperparameters. 
print(f"Best parameters found: {grid_search.best_params_}")

# Train the XGBoost using the best hyperparameters. 
best_xgb_classifier = grid_search.best_estimator_

# Performing the predictions. 
y_pred = best_xgb_classifier.predict(X_test)

# Calculate the accuracy to compare the predictions with the true values. 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the XGBoost Classifier: {accuracy * 100:.2f}%")

#%% F1 Score and MCC. 
# F1 Score Calculation.
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Matthews Correlation Coefficient (MMC).
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

#%% Confussion Matrix. 
# Comparison between “y_pred” and “y_test”.
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=best_xgb_classifier.classes_)

# Plotting.
cmd.plot()
plt.show()

#%% ROC curve. 
# Predict probabilities of each class. 
y_prob = best_xgb_classifier.predict_proba(X_test)[:, 1]

# Calculate the FPR, TPR and the AUC. 
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plotting. 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#%% Precision-recall curve. 
# Calculate the precision and recall. 
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# Calculate the AP. 
avg_precision = average_precision_score(y_test, y_prob)

# Plotting. 
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision)
disp.plot()
plt.title(f'Precision-Recall Curve (Avg Precision = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

