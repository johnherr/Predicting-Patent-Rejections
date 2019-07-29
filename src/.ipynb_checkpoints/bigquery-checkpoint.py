import os
from google.cloud import bigquery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/johnherr/.google_api_key/john_bigquery_key2.json"

def get_data (query, filename):
    '''
    Performs a SQL query on Google Bigquery Databases
    
    Requries: 
        (1) Google account with billing
        (2) Project configured with bigquery api
        
    IMPORTANT:
        Bigquery queries can search huge amounts of data.  Online GUI provides
        query validation and scan size estimate
        
    Args:
        query - str:
            SELECT app_id
            FROM `patents-public-data.uspto_oce_office_actions.office_actions`
            WHERE rejection_101 = "1"
            LIMIT 100
        
        filename: filename to save query as: 
    '''
    
    client = bigquery.Client()

    # Perform a query.
    QUERY = ('''
            SELECT app_id
            FROM `patents-public-data.uspto_oce_office_actions.office_actions`
            WHERE rejection_101 = "1"
            LIMIT 100
            ''')

    df = client.query(QUERY).to_dataframe()  # API request
    df.to_pickle('../data/'+filename)

