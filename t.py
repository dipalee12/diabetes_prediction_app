import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
dataframe = pd.read_csv("diabetes.csv")

# Prepare features and target
X = dataframe.drop(columns="Outcome", axis=1)
Y = dataframe["Outcome"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize classifiers
logic = LogisticRegression()
random = RandomForestClassifier()
decision = DecisionTreeClassifier()
bagging = BaggingClassifier()
boosting = GradientBoostingClassifier()
stacking = StackingClassifier(estimators=[('lr', logic), ('rf', random), ('ab', decision)], final_estimator=logic)

# Voting classifier
estimators = [('lr', bagging), ('rf', boosting), ('ab', stacking)]
eclf = VotingClassifier(estimators=estimators, voting='soft')
params = {'lr__n_estimators': [10, 100]} 
Voting_Classifier_soft = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

# Train the model
Voting_Classifier_soft.fit(X_train, y_train)

# Save the trained model
joblib.dump(Voting_Classifier_soft, 'model.pkl')

# Evaluate the model
y_pred = Voting_Classifier_soft.predict(X_test)
acc_test_lrd_soft = round(Voting_Classifier_soft.score(X_test, y_test) * 100, 2)
recall_lrd_soft = round(recall_score(y_test, y_pred) * 100, 2)

print("Model Accuracy Score:", acc_test_lrd_soft, "%")
print("Model Recall Score:", recall_lrd_soft, "%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
