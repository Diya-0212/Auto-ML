from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate(x,y):
    #split dataset
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    models={
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "KNN":KNeighborsClassifier(),
        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier()
    }

    results={}

    for name,model in  models.items():
        model.fit(x_train,y_train)
        predictions=model.predict(x_test)
        acc=accuracy_score(y_test,predictions)
        results[name]=acc

    return results