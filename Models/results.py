import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

# Create a base class for declarative models
Base = declarative_base()

# Define a model for storing the results of model performance
class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    dataset = Column(String)
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1Score = Column(Float)
    running_time = Column(Integer)
    time_of_execution = Column(DateTime)

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def saveResult(dataset, model_name, accuracy, precision, recall, f1Score, running_time, confusion_matrix):
    # Create an engine that connects to a SQLite database
    engine = create_engine('sqlite:///results.db')

    # Create a session factory
    Session = sessionmaker(bind=engine)
    
    # Create the table in the database
    Base.metadata.create_all(engine)

    # Create a new session
    session = Session()

    # Create a new result object
    result = Result(
        dataset=dataset,
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1Score=f1Score,
        running_time=running_time,
        time_of_execution=datetime.datetime.now()
    )

    # Add the result object to the session
    session.add(result)

    # Commit the transaction
    session.commit()

    # Close the session
    session.close()

    # Normalize the confusion matrix to show percentages instead of counts
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Create a heatmap of the normalized confusion matrix using matplotlib
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'CM: {model_name} - {dataset}')

    # Create the directory if it does not exist
    os.makedirs('CMResults', exist_ok=True)

    # Save the confusion matrix as an image file in the directory
    image_filename = f'./CMResults/{dataset}_{model_name}.png'
    plt.savefig(image_filename)
