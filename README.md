# Titanic-Classification
Predictive model to determine the likelihood of survival for passengers on the Titanic using data science techniques in Python.
Importing Libraries:

Import necessary libraries, including pandas for data manipulation, numpy for numerical operations, seaborn and matplotlib for data visualization, and scikit-learn for machine learning algorithms.
python
Copy code
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
Loading the Dataset:

Load the Titanic dataset into a pandas DataFrame. Ensure that your dataset is in CSV format and replace 'titanic.csv' with the correct file path.
python
Copy code
titanic_data = pd.read_csv('titanic.csv')
Data Exploration and Preprocessing:

Check for missing values in the dataset and handle them. In this case, missing values in the 'Age' column are filled with the median, and missing values in the 'Embarked' column are filled with the mode.
python
Copy code
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
Feature Engineering:

Create new features or modify existing ones to improve the model's predictive performance. Here, features like 'FamilySize' and 'IsAlone' are created based on the number of siblings/spouses and parents/children.
python
Copy code
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data['IsAlone'] = np.where(titanic_data['FamilySize'] == 1, 1, 0)
Convert Categorical Features:

Convert categorical features (like 'Sex' and 'Embarked') into numerical format using one-hot encoding.
python
Copy code
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])
Select Features and Target Variable:

Define the features (X) and the target variable (y) for the model.
python
Copy code
X = titanic_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = titanic_data['Survived']
Train-Test Split:

Split the dataset into training and testing sets to evaluate the model's performance.
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Build a Random Forest Classifier Model:

Create a Random Forest Classifier model and train it using the training data.
python
Copy code
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
Make Predictions and Evaluate the Model:

Use the trained model to make predictions on the test data and evaluate its performance using accuracy, confusion matrix, and classification report.
python
Copy code
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
Feature Importance:

Display the importance of features in the model.
python
Copy code
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()
This is a basic outline, and you may need to adjust the code based on the specifics of your dataset and goals. Additionally, fine-tuning the model and further analysis can be performed based on the insights gained during the exploration phase.
