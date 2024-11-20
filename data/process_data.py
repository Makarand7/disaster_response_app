import sys
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df

def clean_data(df):
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = [category.split('-')[0] for category in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_url):
    engine = create_engine(database_url)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    messages_filepath = os.getenv('MESSAGES_FILEPATH', 'disaster_messages.csv')
    categories_filepath = os.getenv('CATEGORIES_FILEPATH', 'disaster_categories.csv')
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        print("Error: DATABASE_URL not found.")
        sys.exit(1)

    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    save_data(df, database_url)

if __name__ == '__main__':
    main()
