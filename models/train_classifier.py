import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from joblib import dump
import tempfile
import os

def load_data(database_url):
    engine = create_engine(database_url)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    return X, y, y.columns

def build_model():
    pipeline = MultiOutputClassifier(RandomForestClassifier())
    parameters = {'estimator__n_estimators': [50, 100], 'estimator__max_depth': [10, 20]}
    model = GridSearchCV(pipeline, parameters, cv=3, verbose=2, n_jobs=-1)
    return model

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n{classification_report(y_test.iloc[:, i], y_pred[:, i])}')

def save_model(model):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        dump(model, tmp_file.name)
        print(f"Model saved temporarily at {tmp_file.name}")

def main():
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        print("Error: DATABASE_URL not found.")
        sys.exit(1)

    X, y, category_names = load_data(database_url)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, category_names)
    save_model(model)

if __name__ == '__main__':
    main()
