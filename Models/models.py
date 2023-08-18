from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import datetime
import time

from results import saveResult 

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from hmmlearn import hmm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

def transformText(dataset):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['TEXT'], dataset['CLASS'], test_size=0.2)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training and test sets
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test

def visualize_vectors(vectors, n_clusters=3):
    # Reduce the dimensionality of the vectors to 2 using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors.toarray())

    # Cluster the reduced vectors using KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_vectors)
    labels = kmeans.labels_

    # Plot the reduced vectors on a scatter plot
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis')
    plt.show()


def runNaiveBayes(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)

    visualize_vectors(X_train)

    # Create a Naive Bayes classifier
    clf = MultinomialNB()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Naive Bayes',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runSVM(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create an SVM classifier
    clf = SVC()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='SVM',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runHMM(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Convert the training and test data from sparse matrices to dense numpy arrays
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Create an HMM classifier
    clf = hmm.GaussianHMM()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test,y_pred ,average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='HMM',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runGradientBoosting(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create a Gradient Boosting classifier
    clf = GradientBoostingClassifier()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Gradient Boosting',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runRandomForest(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create a Random Forest classifier
    clf = RandomForestClassifier()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test,y_pred ,average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Random Forest',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runLogisticRegression(dataset, name):
    
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create a Logistic Regression classifier
    clf = LogisticRegression()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test,y_pred ,average='macro')
    f1Score=f1_score(y_test,y_pred ,average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Logistic Regression',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runDecisionTree(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier()

    # Record the start time
    start_time = time.time()

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    # Make predictions on the test set using the classifier
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the performance metrics using macro averaging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    # Save the results using the save_results function
    saveResult(
        dataset=name,
        model_name='Decision Tree',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runKNN(dataset,name):
    
    X_train,X_test,y_train,y_test=transformText(dataset)
    
    clf=KNeighborsClassifier()
    
    start_time=time.time()
    
    clf.fit(X_train,y_train)
    
    end_time=time.time()
    
    running_time=end_time-start_time
    
    y_pred=clf.predict(X_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy=accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    saveResult(
        dataset=name,
        model_name='KNN',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runXGBoost(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)
    
    clf = XGBClassifier()
    
    start_time = time.time()
    
    clf.fit(X_train, y_train)
    
    end_time = time.time()
    
    running_time = end_time - start_time
    
    y_pred = clf.predict(X_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    saveResult(
        dataset=name,
        model_name='XGBoost',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

def runAdaBoost(dataset, name):
    X_train, X_test, y_train, y_test = transformText(dataset)
    
    clf = AdaBoostClassifier()
    
    start_time = time.time()
    
    clf.fit(X_train, y_train)
    
    end_time = time.time()
    
    running_time = end_time - start_time
    
    y_pred = clf.predict(X_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')

    saveResult(
        dataset=name,
        model_name='AdaBoost',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        confusion_matrix=cm  # Pass the confusion matrix as an argument
    )

