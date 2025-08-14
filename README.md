# Reddit_Fake_News_Detection

## Project Introduction

This project is part of the Kaggle competition DEPI-R2-Competition1, which focuses on detecting fake news in Reddit posts. Social media platforms like Reddit have become major sources of information, but the rise of misinformation—especially in politics, business, and public health—has made automated detection essential. The goal is to develop AI-driven text classification models that can accurately predict whether a post is fake news (1) or genuine (0) using both text and metadata. 
Data set used : https://www.kaggle.com/competitions/depi-r-2-competition-1/data 

## Model Pipelines & Results
### Pipeline 1 – Classic Machine Learning with TF-IDF

- **Approach:** Preprocessed text (lowercasing, punctuation & stopword removal, lemmatization) and vectorized with TF-IDF (unigrams + bigrams).
- **Models Tested:** Logistic Regression, Linear SVM, and SGDClassifier.
- **Performance:** These models were computationally efficient and provided strong baselines. Logistic Regression and Linear SVM achieved competitive F1-scores. 

### Pipeline 2 – Word2Vec Embeddings + Classifiers

- **Approach:** Used pretrained Word2Vec embeddings via gensim, averaged token vectors for each document.
- **Models Tested:** Logistic Regression, Linear SVM, XGBoost.
- **Performance:** Captured semantic relationships between words better than TF-IDF, improving recall on difficult cases. XGBoost performed the best in this pipeline, leveraging embedding features effectively for non-linear decision boundaries.

### Pipeline 3 – FastText Embeddings + Clustering (Unsupervised)

- **Approach:** Averaged FastText embeddings, then applied KMeans clustering into two groups. Mapped clusters to classes using majority voting.
- **Performance:** As an unsupervised approach, it was notably less accurate than supervised methods. While it captured some structure in the data, misclassifications were common, showing the challenge of fake news detection without label guidance.

### Pipeline 4 – Neural Network (Kaggle Solution Adaptation)

- **Approach:** Adapted a Kaggle-provided Dense Neural Network architecture (Embedding -> Flatten -> Dense layers with dropout).
- **Performance:** Achieved the highest F1-score across all pipelines. The learned embeddings combined with nonlinear transformations allowed the model to capture nuanced linguistic patterns. Early stopping and learning rate scheduling improved convergence and reduced overfitting.

### Conclusion:
Among all pipelines, the **Neural Network** delivered the best results, confirming that deep learning architectures can outperform traditional methods in fake news detection—especially when trained on well-preprocessed data. However, TF-IDF + Logistic Regression/SVM remains a strong, interpretable, and fast baseline for production scenarios with limited computation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ccd99405-cb7d-4ae0-9cf2-f75fe90515fd" alt="Heatmap DNN" width="45%" />
  <img src="https://github.com/user-attachments/assets/7c066da3-0934-4c81-8b6a-2b56da897144" alt="Heatmap  SVM" width="44%" />
</p>


