
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

data = pd.read_csv("data/diabetes_clean.csv")

x = data.drop("Outcome", axis=1)
y = data["Outcome"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=43,stratify=y)

model =  Pipeline([
    ('smote', SMOTE(random_state=45)),
    ('model', AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=200,
    learning_rate=0.1,
    random_state=25
)
)
])


model.fit(x_train, y_train)

y_predict_adaboost=model.predict(x_test)

print("Adaboost testing score : ",model.score(x_test,y_test))
print("Adaboost trainig score",model.score(x_train,y_train))

from sklearn.metrics import classification_report, confusion_matrix

print("Adaboost confusion metric score : \n" ,confusion_matrix(y_test, y_predict_adaboost))
print("Adaboost classification report : \n",classification_report(y_test, y_predict_adaboost))



joblib.dump(model, "models/adaboost_model.pkl")

print("Model trained and saved successfully.")