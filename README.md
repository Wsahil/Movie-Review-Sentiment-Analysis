# Movie-Review-Sentiment-Analysis
This project aimed to leverage natural language processing (NLP) and machine learning techniques to perform sentiment analysis on a large dataset of customer movie reviews. By extracting insights from customer opinions and preferences, the project sought to predict sentiment polarity effectively.

## Project Overview

- **Dataset**: Rotten Tomatoes dataset with approximately 156,060 movie review phrases and corresponding sentiment labels (negative, somewhat negative, neutral, somewhat positive, positive).
  ![image](https://github.com/Wsahil/Movie-Review-Sentiment-Analysis/assets/71370836/887838ab-72ae-442c-8286-100d622a4f86)

- **Technologies Used**: Python (NLTK, Scikit-learn), Natural Language Processing, Machine Learning Algorithms (Logistic Regression, Naive Bayes, SVM, Random Forest)
- **Key Objectives**:
  - Preprocess the dataset and create diverse feature sets.
  - Develop machine learning models for sentiment analysis and prediction.
  - Engineer feature functions like sentiment lexicons and unigrams.
  - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

## Project Implementation

1. **Data Preprocessing**: Performed tokenization, case conversion, stop-word removal, non-alphabetic character removal, and stemming on the movie review phrases to prepare the data for feature extraction and modeling.

2. **Feature Engineering**: Created various feature sets, including:
   - Bag of Words (BoW)
   - Unigrams
   - Sentiment Lexicons (SL)
   - Part-of-Speech (POS) tags
   - Linguistic Inquiry and Word Count (LIWC)
   - Combined features (SL-LIWC)

3. **Model Development**: Trained and evaluated multiple machine learning models for sentiment analysis, including:
   - Naive Bayes (NLTK)
   - Support Vector Machine (SVM) (Scikit-learn)
   - Random Forest (Scikit-learn)

4. **Cross-Validation**: Implemented cross-validation techniques, specifically for the Random Forest classifier, to evaluate model robustness and generalization performance.

5. **Evaluation Metrics**: Calculated evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess the performance of the trained models.
Summary of Accuracies:
![image](https://github.com/Wsahil/Movie-Review-Sentiment-Analysis/assets/71370836/c032b494-e28c-4785-a05b-fcccb2e1c7d6)

## Key Findings and Results

- The LIWC feature set achieved the highest accuracy of 55% in predicting sentiment polarity of movie reviews using the Naive Bayes classifier.
- Combining LIWC and Sentiment Lexicon (SL) features with Naive Bayes yielded the second-highest accuracy of 54.33%.
- Cross-validation with the Random Forest classifier did not significantly improve model efficiency in classifying movie reviews.

## Conclusion

This project demonstrated the application of natural language processing and machine learning techniques for sentiment analysis on movie reviews. While the achieved accuracy of 55% is promising, there is potential for further improvement through techniques like hyperparameter tuning and ensemble methods. The insights gained from this project can contribute to a better understanding of customer preferences and support informed decision-making in the entertainment industry.
