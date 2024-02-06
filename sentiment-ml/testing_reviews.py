import random
import numpy as np

def TestingReviews(test_data, amount:int=5) -> list:
    '''
    This function will ingest a list of reviews and run the model to predict the reviewer's sentiment.

    It will take 5 reviews by default, but you can pass any number you want.
    '''
    list_rand = []
    limit = len(test_data)
    rand_row = random.randrange(amount, limit, 1)
    for n in range(amount):
        rand_row = random.randrange(0, limit, 1)
        row_to_test = [test_data.user_review[rand_row]]
        row_test = vectorizer.transform(row_to_test)
        row_prediction = clf_svc.predict(row_test)[0]
        row_proba = clf_svc.predict_proba(row_test)
        row_max_proba = round(np.max(row_proba) * 100, 2)
        print(f'Review: {row_to_test}')
        print(f'Sentiment: {row_prediction} - Confidence: {row_max_proba}%')