from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
import time

from results import saveResult 

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from hmmlearn import hmm

def transformText(dataset):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['TEXT'], dataset['CLASS'], test_size=0.2)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training and test sets
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test

def runNaiveBayes(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create a Naive Bayes classifier
    clf = MultinomialNB()

    # Define the hyperparameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    }

    # Create a Grid Search object
    grid_search = GridSearchCV(clf, param_grid)

    # Record the start time
    start_time = time.time()

    # Fit the Grid Search object to the training data
    grid_search.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Get the best estimator from the Grid Search
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best estimator
    y_pred = best_clf.predict(X_test)

    # Calculate the performance metrics using weighted averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1Score = f1_score(y_test, y_pred, average='weighted')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Naive Bayes',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
    )

def runSVM(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create an SVM classifier
    clf = SVC()

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }

    # Create a Grid Search object
    grid_search = GridSearchCV(clf, param_grid)

    # Record the start time
    start_time = time.time()

    # Fit the Grid Search object to the training data
    grid_search.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Get the best estimator from the Grid Search
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best estimator
    y_pred = best_clf.predict(X_test)

    # Calculate the performance metrics using weighted averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1Score = f1_score(y_test, y_pred, average='weighted')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='SVM',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
    )
    
def runHMM(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Convert the training and test data from sparse matrices to dense numpy arrays
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Create an HMM classifier
    clf = hmm.GaussianHMM()

    # Define the hyperparameter grid
    param_grid = {
        'n_components': [2, 3, 4],
        'covariance_type': ['spherical', 'diag', 'tied', 'full']
    }

    # Create a Grid Search object
    grid_search = GridSearchCV(clf, param_grid)

    # Record the start time
    start_time = time.time()

    # Fit the Grid Search object to the training data
    grid_search.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Get the best estimator from the Grid Search
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best estimator
    y_pred = best_clf.predict(X_test)

    # Calculate the performance metrics using weighted averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1Score = f1_score(y_test, y_pred, average='weighted')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='HMM',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
    )
