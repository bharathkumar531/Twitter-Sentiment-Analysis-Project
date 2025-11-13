# Twitter-Sentiment-Analysis-Project
#  Twitter Sentiment Analysis (Sentiment140 Dataset)
This project performs sentiment classification on 1.6 million tweets using the Sentiment140 dataset.  
It includes preprocessing, TF-IDF vectorization, and multiple ML models to classify tweets as Positive or Negative

#Project Overview
Loaded and cleaned the Sentiment140 dataset  
Converted numeric sentiment labels   
Applied TF-IDF vectorization  
Trained Naive Bayes, Logistic Regression, and Linear SVM  
Evaluated model performance  
Predicted sentiment for new tweets  
---

##  Dataset – Sentiment140

The Sentiment140 dataset consists of 1.6M tweets with the following structure:

| Column     | Description                        |
|------------|------------------------------------|
| sentiment  | 0 = Negative, 4 = Positive         |
| id         | Tweet ID                           |
| date       | Timestamp                          |
| query      | Query (unused)                     |
| user       | Username                           |
| text       | Tweet text                         |


# Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TF-IDF  
- Jupyter Notebook  

---

##  Project Structure

```
Twitter-Sentiment-Analysis-Project/
│
├── sentiment_analysis.ipynb     # Main notebook
├── README.md                    # Documentation

## 🔧 Installation

Install Python dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
sentiment_analysis.ipynb
```

---

##  Data Preprocessing

Main cleaning steps:

- Removing unwanted columns  
- Handling null values  
- Converting labels  
- TF-IDF conversion  

```python
df['text'] = df['text'].fillna('')
df = df.dropna(subset=['sentiment'])

##  Models Used

- Bernoulli Naive Bayes  
- Logistic Regression  
- Linear SVM (best performing model)  

## Model Accuracy

Update these values based on your output:

| Model                   | Accuracy |
|-------------------------|----------|
| Bernoulli Naive Bayes   | 78%      |
| Logistic Regression      | 78%      |
| Linear SVM              | 79% (best) |

---

##  Predicting New Tweets

```python
sample = ["I love this!", "This is terrible."]
sample_vec = tfidf.transform(sample)
svm.predict(sample_vec)

## Future Improvements

- Add deep learning models (BERT)  
- Deploy as Streamlit web app  
- Advanced tweet cleaning (hashtags, URLs, mentions)  
- Hyperparameter tuning  

## Author

Bharath Kumar  
GitHub: https://github.com/bharathkumar531
