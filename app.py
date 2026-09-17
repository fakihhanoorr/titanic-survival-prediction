
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("titanic.csv.csv")

# Select useful columns
data = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
].copy()

# Fill missing values
data["Age"] = data["Age"].fillna(data["Age"].median())

data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

# Convert text into numbers
data["Sex"] = data["Sex"].map({
    "male": 0,
    "female": 1
})

data["Embarked"] = data["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Prepare data
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# APP UI
# -----------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger information to predict "
    "whether the passenger is likely to survive."
)

st.divider()

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sibsp = st.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

st.divider()

if st.button("Predict Survival"):

    sex_value = 1 if sex == "female" else 0

    embarked_values = {
        "S": 0,
        "C": 1,
        "Q": 2
    }

    embarked_value = embarked_values[embarked]

    passenger = pd.DataFrame(
        [[
            pclass,
            sex_value,
            age,
            sibsp,
            parch,
            fare,
            embarked_value
        ]],
        columns=X.columns
    )

    prediction = model.predict(passenger)[0]

    probability = model.predict_proba(passenger)[0]

    if prediction == 1:

        st.success(
            "Prediction: Passenger is predicted to SURVIVE."
        )

        st.write(
            f"Survival probability: "
            f"{probability[1] * 100:.2f}%"
        )

    else:

        st.error(
            "Prediction: Passenger is predicted to NOT SURVIVE."
        )

        st.write(
            f"Survival probability: "
            f"{probability[1] * 100:.2f}%"
        )

st.divider()

st.caption(
    "Titanic Survival Prediction | Machine Learning Project"
)
