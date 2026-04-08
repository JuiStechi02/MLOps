import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, plot_importance


df = pd.read_csv("titanic.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())

df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_child'] = (df['Age'] < 16).astype(int)


df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'is_child']]



X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", df.shape)
print("Class distribution:\n", df['Survived'].value_counts())
print("Correlation with Survived:\n", df.corr()['Survived'].sort_values())
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
model = XGBClassifier(n_estimators=50, max_depth=2)
model.fit(X_train, y_train)


pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)


cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("plot.png")


plot_importance(model)
plt.title("Feature importance")
plt.savefig("plot2.png")

