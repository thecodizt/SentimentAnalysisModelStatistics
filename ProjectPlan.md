# Statistical Analysis of Machine Learning Models for Text Classification and Sentiment Analysis in Social Media Messages

## Datasets

Dataset | Size | Features | Label Field | Link | 
| --- | --- | --- | --- | --- | 
| Stanford Sentiment Treebank | 215,154 unique phrases | Sentiment labels, parse trees | Sentiment labels | [Link](https://nlp.stanford.edu/sentiment/treebank.html) | 
| Amazon Product Data | 142.8 million reviews | Reviews, product metadata, links | Star ratings | [Link](http://jmcauley.ucsd.edu/data/amazon/) | 
| Multi-Domain Sentiment Dataset | Varies by domain | Product reviews, star ratings | Star ratings | [Link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) | 
| IMDB Movie Reviews Dataset | 50,000 reviews | Movie reviews, binary sentiment labels | Binary sentiment labels | [Link](http://ai.stanford.edu/~amaas/data/sentiment/) |
| Sentiment140 | 1.6 million tweets | Tweet text, polarity labels | Polarity labels | [Link](http://help.sentiment140.com/for-students) |
| Newsdata.io news dataset | Varies by query | News articles, metadata | N/A | [Link](https://newsdata.io/) |
| SES: Sentiment Elicitation System for Social Media Data| N/A| N/A| N/A| [Link](https://cucis.eecs.northwestern.edu/projects/DMS/publications.html)|
| Mining Millions of Reviews: A Technique to Rank Products Based on Importance of Reviews| N/A| N/A| N/A|[Link](https://cucis.eecs.northwestern.edu/projects/DMS/publications.html)|
| Twitter US Airline Sentiment Dataset| 14,640 instances| Tweet text, sentiment labels, negative reasons| Sentiment labels|[Link](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)|
| Yelp Dataset| 6,990,280 reviews| Reviews, business information, user information, check-ins| Star ratings|[Link](https://www.yelp.com/dataset)|
| Sentiment Analysis Dataset on Kaggle| N/A| Tweet text, polarity labels| Polarity labels|[Link](https://www.kaggle.com/kazanova/sentiment140)|
| Stock Sentiment Analysis Dataset| 1000 discussions| Social media discussions of publicly traded stocks| Positive or negative sentiment associated with each discussion|[Link](https://www.kaggle.com/yash612/stockmarket-sentiment-dataset)|
| Webis-CLS-10 Dataset| 800,000 Amazon product reviews|^13^Product reviews in English, German, French and Japanese languages|^13^N/A|^13^[Link](https://webis.de/data/webis-cls-10.html)|


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
