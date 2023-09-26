import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
file_path = 'Titanic.csv'
df_s = pd.read_csv(file_path)
df = df_s

# Handle missing values
df['age'].fillna(df['age'].mean(), inplace=True) # mean for null age fields
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True) # mode for null embarked fields

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'embarked'])

# Convert boolean columns to integers
bool_columns = ['gender_F', 'gender_M', 'embarked_Cherbourg', 'embarked_Queenstown', 'embarked_Southampton']  # Adjust these names based on the output from step 2
for col in bool_columns:
    df[col] = df[col].astype(int)

# Split the data
df = df[['age', 'class', 'fare', 'survived', 'gender_F',
       'gender_M', 'embarked_Cherbourg', 'embarked_Queenstown',
       'embarked_Southampton']]
X = df.drop('survived', axis=1)
y = df['survived']

# train test splitting and for the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# streamlit start
st.title('Titanic Survival Prediction')
st.dataframe(df_s)
st.write('Using Logistic Regression')
st.write(f'Accuracy: {accuracy * 100:.2f}%')

# User input
gender = st.selectbox('Gender:', ('male', 'female'))
age = st.number_input('Age:')
pclass = st.selectbox('Class:', (1, 2, 3))
fare = st.number_input('Fare:')
embarked = st.selectbox('Embarked:', ('C', 'Q', 'S'))

if st.button('Predict'):
    # Convert to one-hot encoding
    user_input = [[age, pclass, fare, 
                   1 if gender == 'male' else 0, 0 if gender == 'male' else 1,
                   1 if embarked == 'C' else 0, 1 if embarked == 'Q' else 0, 1 if embarked == 'S' else 0]]

    prediction = model.predict(user_input)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.write(f'The predicted result is: {result}')