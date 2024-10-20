import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
from sklearn.metrics import f1_score, matthews_corrcoef 

#%% Importing data.
data = pd.read_csv('/Users/rodrigosanchez/Documents/ScientificProgramming/Experiment.csv')

#%% Random Forest. 
# Data frame with the feature selection.  
X = data[['ContrastHeterogeneity', 'GridCoarseness']]

# Define the target variable “y”, that contains the binary values. 
y = data['Correct']

# Split the data set into training (70%) and test (30%), stratify the “y” data set to balance the data from the “y_train”.  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

# Train the Random Forest classifier with the number of decision trees are 100 (n_estimators = 100). 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Performing the predictions. 
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy to compare the predictions with the true values. 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy * 100:.2f}%")

#%% F1 score and the Matthew Correlation Coefficient. 
# F1 Score Calculation. 
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Matthews Correlation Coefficient (MMC). 
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

#%% Confusion Matrix. 
# Comparison between “y_pred” and “y_test”. 
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=rf_classifier.classes_)

# Plotting. 
cmd.plot()
plt.show()

#%% Feature importance. 
# Extract the feature importance from the trained model. 
importances = rf_classifier.feature_importances_
features = X.columns

# Plotting. 
plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.show()

#%% ROC curve. 
# Predict probabilities of each class. 
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate the FPR, TPR and the AUC. 
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting. 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='Blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#%% Precision-Recall Curve. 
# Predict probabilities. 
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

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