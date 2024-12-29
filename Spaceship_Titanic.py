import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer

data = pd.read_csv(r"/Mechine Learning/Dataset/Spaceship Titanic/train.csv")

# Processing HomePlanet column.
def cleanID(ID):
    return ID[0:4]
data["PassengerId"] = data["PassengerId"].apply(cleanID)
def HP_clean(data, ID, HP):
    for i in range(len(data) - 1):
        if data[ID].iloc[i] == data[ID].iloc[i + 1]:
            if pd.isna(data[HP].iloc[i]):
                data.at[i, HP] = data[HP].iloc[i + 1]
            elif pd.isna(data[HP].iloc[i + 1]):
                data.at[i + 1, HP] = data[HP].iloc[i]
HP_clean(data, "PassengerId", "HomePlanet")
imputer_HP = SimpleImputer(strategy="most_frequent")
data["HomePlanet"] = imputer_HP.fit_transform(data[["HomePlanet"]]).ravel()



# Processing CryoSleep column.
data["Spending"] = data[["RoomService", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
def CS_clean(data, CS, S):
    for i in range(len(data) - 1):
        if pd.isna(data[CS].iloc[i]):
            if data[S].iloc[i] > 0:
                data.at[i, CS] = False
            elif data[S].iloc[i] == 0:
                data.at[i, CS] = True
CS_clean(data, "CryoSleep", "Spending")



# Processing Destination column.
def Des_clean(data, ID, Des):
    for i in range(len(data) - 1):
        if data[ID].iloc[i] == data[ID].iloc[i + 1]:
            if pd.isna(data[Des].iloc[i]):
                data.at[i, Des] = data[Des].iloc[i + 1]
            elif pd.isna(data[Des].iloc[i + 1]):
                data.at[i + 1, Des] = data[Des].iloc[i]
Des_clean(data, "PassengerId", "Destination")
imputer_Des = SimpleImputer(strategy="most_frequent")
data["Destination"] = imputer_Des.fit_transform(data[["Destination"]]).ravel()



# Processing Age column.
imputer_Age = SimpleImputer(strategy="median")
data["Age"] = imputer_Age.fit_transform(data[["Age"]])



# Processing VIP column.
imputer_VIP = SimpleImputer(strategy="most_frequent")
data["VIP"] = imputer_VIP.fit_transform(data[["VIP"]]).ravel()



x = data.drop(["PassengerId", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name", "Transported", "Cabin"], axis=1)
y = data["Transported"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Age, Spending.
num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# HomePlanet, Destination.
nom_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder())
])

# CryoSleep.
ord_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["Age", "Spending"]),
    ("nom_features", nom_transformer, ["HomePlanet", "Destination"]),
    ("ord_features", ord_transformer, ["CryoSleep","VIP"]),
])

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42, loss='log_loss'))
])


param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, 9],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.05],
    'model__subsample': [0.8, 0.9, 1.0]
}
# Best parameters found:  {'model__learning_rate': 0.01, 'model__max_depth': 3, 'model__n_estimators': 100, 'model__subsample': 0.8}
grid_search = GridSearchCV(clf, param_grid = param_grid , cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(x_train, y_train)


best_model = grid_search.best_estimator_
y_predict = best_model.predict(x_test)


print("Best parameters found: ", grid_search.best_params_)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test, y_predict))



