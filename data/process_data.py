import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load dataframe from sources
    INPUT
    messages_filepath
    categories_filepath
    OUTPUT
    df - pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df

def clean_data(df):
    """
    Data cleaning and transform the categories columns
    INPUT
    pandas DataFrame
    OUTPUT
    cleaned pandas DataFrame with columns transformed
    
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[[0]]
    category_colnames = [i.split('-')[0] for i in row.values[0]]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], join='inner', axis=1)
    df.drop_duplicates(inplace = True)
    print('Duplicates remaining:', df.duplicated().sum())
    return df

def save_data(df, database_filename):
    """
    Saves DataFrame to the database file path
    """
    filename = 'sqlite:///' + database_filename
    engine = create_engine(filename)
    df.to_sql('disaster_response_df', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()