import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    """ Load data and clean up
    INPUT
    filepath of the database
    OUTPUT
    X - explanatory matrix
    Y - response matrix
    categories - categories of the events
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql('SELECT * FROM disaster_response_df', engine)
    X = df['message']
    y = df[df.columns[5:]]
    categories = y.columns
    return X, y, categories

def tokenize(text):
    """ tokenize the text
    INPUT
    a text string
    OUTPUT
    tokenized and transformed text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """ build a grid search model
    INPUT
    None
    OUTPUT
    A model with classifier and pipeline
    """
    moc = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
    ])
    parameters = {
        'clf__estimator__max_depth': [10, 50, None]
    }
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Print model results for evaluation
    INPUT
    model -- estimator-object
    X_test -- test data X
    y_test -- test data y
    category_names -- list of catgories (string)
    OUTPUT
    none
    """
    Y_pred = model.predict(X_test)
    results = pd.DataFrame(columns = ['category','precision', 'recall', 'f1_score'])
    for i, col in enumerate(Y_test.columns):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[col], Y_pred[:, i], average = 'weighted')
        results.at[i, 'category'] = col
        results.at[i, 'precision'] = precision
        results.at[i, 'recall'] = recall
        results.at[i, 'f1_score'] = fscore
    print('Aggregated f_score:', results['f1_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())

def save_model(model, model_filepath):
    """Save model as pickle file
    INPUT
    fitted model
    file path in which the model is saved at
    OUTPUT
    none
    """
    outfile = open(model_filepath, 'wb')
    pickle.dump(model, outfile)
    outfile.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()