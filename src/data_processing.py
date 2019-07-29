# import os
# from google.cloud import bigquery
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/johnherr/.google_api_key/john_bigquery_key2.json"
# import re
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

def get_data (SQL_query, filename):
    '''Performs a SQL query on Google BigQuery Databases
    REQUIRES:
        (1) Google account with billing
        (2) Project in BigQuery configured with BigQuery api
    IMPORTANT:
        Bigquery queries can search huge amounts of data.  Online GUI provides
        query validation and scan size estimate
    ARGS:
        SQL_query - str:
            e.g.
            "SELECT app_id
            FROM `patents-public-data.uspto_oce_office_actions.office_actions`
            WHERE rejection_101 = "1"
            LIMIT 100"
        filename: filename to save query as: '''

    client = bigquery.Client()
    df = client.query(SQL_query).to_dataframe()  # API request
    df.to_pickle(filename)

def open_saved_pickle(name, filed_col = 'filed_claims', granted_col = 'granted_claims'):
    ''' Opens pickeled pandas df and applies Regex command to exctract claim 1

    ARGS:
        Name: Filepath of df
        filed_col: columns with raw text from of filed application
        granted_col: columns with raw text from granted application '''
    df = pd.read_pickle(name)
    df = get_first_claim(df, filed_col)
    df = get_first_claim(df, granted_col)
    df = df.dropna()
    return df

def get_first_claim(df, col):
    ''' Uses regex to select only claim 1 from a string cointaing all raw text data
    ARGS: pandas DataFrame
    RETURNS: pandas DataFrame'''
    df[col] = df[col].apply(lambda x: regex_claim(x))
    return df

def regex_claim(string):
    ''' Helper function for applying Regex to extract first claim'''
    pattern = re.compile(r'1\s?\.\s*A([^2])*')
    try:
        result = pattern.search(string)[0]
    except:
        result = None
    return result

def split_train_test(df, test_size=.20):
    '''Split pandas DataFrame into train and test sets.
    ARGS:
        df: Pandas DF, each row has a rejected & allowed claim
        test_size: fraction reserved for test
    RETURNS:
        train/test dataframes'''
    msk = np.random.rand(len(df)) < (1-test_size)
    train = split_filed_granted(df[msk])
    test = split_filed_granted(df[~msk])

    return train, test

def split_filed_granted(df):
    '''Split each row. New rows have either a rejected claim or an accepted claim
    ARGS:
        df: pandas df.
    '''
    df_filed = df[['filed_claims','app_id','art_unit','uspc_class']].copy()
    df_filed['rejected']=1
    df_filed['allowed']=0
    df_granted = df[['granted_claims','app_id','art_unit','uspc_class']].copy()
    df_granted['rejected']=0
    df_granted['allowed']=1
    all_dfs = [df_filed, df_granted]
    for _ in all_dfs:     # Give all df's common column names
        _.columns = ['claim','app_id','art_unit','uspc_class', 'rejected','allowed']
    new_df = pd.concat(all_dfs).reset_index(drop=True)
    return new_df

def infer_claim_vectors(df, model_fp='models/doc2vec/enwiki_dbow/doc2vec.bin',
                  claims_col='claim', vector_col='claim_vec'):
    '''Vectorizes Claim text using pretrained model. If the model has been
    trained on the claim text in the training set, then vectors only need
    to be infered for the test set_title
    ARGS:
        model_fp:filepath to doc2vec models
        df: pandas df
        claims_col: column name with raw claim get_texts
        vector_col: name of column to store vectorized claims
    RETURN:
        df
    '''
    model = Doc2Vec.load('models/doc2vec/enwiki_dbow/doc2vec.bin')
    vectors = [model.infer_vector(claim.strip().split(" "))
                for claim in df[claims_col]]
    df[vector_col] = vectors
    return df
