import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("insurance.csv")


data["sex"] = data["sex"].map({"male": 0, "female": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data["region"] = data["region"].map({
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
})

X = data.drop("charges", axis=1)
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")

score = model.score(X_test, y_test)
print("Model Accuracy:", score)