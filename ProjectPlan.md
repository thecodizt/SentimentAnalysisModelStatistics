<p><a target="_blank" href="https://app.eraser.io/workspace/kPrQ6JQHz5C8prYF90mp" id="edit-in-eraser-github-link"><img alt="Edit in Eraser" src="https://firebasestorage.googleapis.com/v0/b/second-petal-295822.appspot.com/o/images%2Fgithub%2FOpen%20in%20Eraser.svg?alt=media&amp;token=968381c8-a7e7-472a-8ed6-4a6626da5501"></a></p>

# Statistical Analysis of Machine Learning Models for Text Classification and Sentiment Analysis in Social Media Messages


![Workflow](/.eraser/kPrQ6JQHz5C8prYF90mp___NSX35knPbzTDJN8ATbww765SbPq2___---figure---cDuEHRqfL0zeXrqh4qoPK---figure---t-Uwyplr5444steHVcCy5g.png "Workflow")

## Datasets
Dataset

Size

Features

Label Field

Link

IMDB Movie Reviews Dataset

50,000 reviews

Movie reviews, binary sentiment labels

Binary sentiment labels

[﻿Link](http://ai.stanford.edu/~amaas/data/sentiment/) 

Sentiment140

1.6 million tweets

Tweet text, polarity labels

Polarity labels

[﻿Link](http://help.sentiment140.com/for-students) 

Twitter US Airline Sentiment Dataset

14,640 instances

Tweet text, sentiment labels, negative reasons

Sentiment labels

[﻿Link](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) 

UTKML Twitter Spam Detection Competition Dataset

N/A

Tweet text, user information, tweet metadata

Binary spam labels

[﻿Link](https://www.kaggle.com/c/utkmls-twitter-spam-detection-competition) 

Hate Speech and Offensive Language Dataset (DONE)

24,783 tweets

Tweet text, class labels, confidence scores

Class labels (hate speech, offensive language, neither)

[﻿Link](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) 

For the common dataset evaluation by the models, the expected struture of the datasets is [Message, Score]

### TODO: Decide final list of datasets to be used
## Models
There are many machine learning and deep learning models that can be used for text classification. Some popular machine learning models for text classification include:

- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Logistic Regression**
- **Decision Trees**
- **Random Forests**
Some popular deep learning models for text classification include:

- **Convolutional Neural Networks (CNN)**
- **Recurrent Neural Networks (RNN)**
- **XLNet**
- **ERNIE**
- **Text-to-Text Transfer Transformer (T5)**
- **Binary Partitioning Transfomer (BPT)**
- **Neural Attentive Bag-of-Entities (NABoE)**
## Testing strategy
- **Accuracy**: This measures the proportion of correctly classified instances out of the total number of instances. It is a commonly used metric for evaluating the performance of classification models.
- **Precision**: This measures the proportion of true positive instances among the instances that were classified as positive by the model. It is a measure of how well the model identifies positive instances.
- **Recall**: This measures the proportion of true positive instances among all positive instances in the dataset. It is a measure of how well the model identifies all positive instances.
- **F1-score**: This is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance in terms of both precision and recall.
- **Confusion matrix**: This is a table that shows the number of true positive, false positive, true negative, and false negative predictions made by the model. It provides a detailed view of the model's performance.
### DOUBT: Best metric? Should we go for multiple metrics? Which of these?
- **t-test**: This test can be used to compare the means of two groups, such as the performance of two different models on the same dataset. The t-test assumes that the data is normally distributed and that the variances of the two groups are equal⁴.
- **Wilcoxon signed-rank test**: This is a non-parametric test that can be used as an alternative to the t-test when the data is not normally distributed or when the sample size is small⁴.
- **ANOVA (Analysis of Variance)**: This test can be used to compare the means of more than two groups, such as the performance of multiple models on the same dataset. ANOVA assumes that the data is normally distributed and that the variances of the groups are equal⁴.
- **Kruskal-Wallis test**: This is a non-parametric test that can be used as an alternative to ANOVA when the data is not normally distributed or when the sample size is small⁴.
### DOUBT: Which statistical test to apply? Or again multiple tests?



<!--- Eraser file: https://app.eraser.io/workspace/kPrQ6JQHz5C8prYF90mp --->
