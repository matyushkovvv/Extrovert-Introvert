from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


# load data
data = pd.read_csv('./dataset/personality_dataset.csv')

# prepare data
data.dropna(inplace=True)
data['Stage_fear'] = data['Stage_fear'].replace({'Yes': 1, 'No': 0}).astype(int)
data['Drained_after_socializing'] = data['Drained_after_socializing'].replace({'Yes': 1, 'No': 0}).astype(int)
X = data.drop('Personality', axis=1)
Y = data['Personality']

# split data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

models = [
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42))
]

for name, model in models:
    model.fit(X_Train, Y_Train)
    print(model.score(X_Test, Y_Test))

    accuracy = model.score(X_Test, Y_Test)
    print(f'\n{name} Accuracy: {accuracy:.2f}')
    
    Y_Pred = model.predict(X_Test)
    
    print(f'\nClassification Report for {name}:')
    print(classification_report(Y_Test, Y_Pred))
    
    cm = confusion_matrix(Y_Test, Y_Pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()