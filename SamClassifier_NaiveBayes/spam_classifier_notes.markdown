# Spam Email Classifier with Naive Bayes: Reference Notes

## 1. Naive Bayes Overview
- **What is it?**: A probabilistic algorithm based on Bayes’ Theorem for classification, ideal for text tasks like spam detection.
- **Bayes’ Theorem**:
  \[ P(\text{Spam} | \text{Words}) = \frac{P(\text{Words} | \text{Spam}) \cdot P(\text{Spam})}{P(\text{Words})} \]
  - \( P(\text{Spam} | \text{Words}) \): Probability email is spam given its words.
  - \( P(\text{Words} | \text{Spam}) \): Likelihood of words in spam emails.
  - \( P(\text{Spam}) \): Prior probability of spam (e.g., 0.5 if half the emails are spam).
  - \( P(\text{Words}) \): Normalizing factor (same for spam and ham).
- **Naive Assumption**: Words are independent (e.g., “free” and “click” don’t affect each other’s probabilities).
- **Why for Spam?**:
  - Fast and handles high-dimensional data (many words).
  - Effective at catching spam patterns (e.g., “free”, “win”).
- **Classes**: Spam (1), Ham (0).

## 2. Implementation Steps
Based on the Python code for a spam email classifier using a small dataset, NLTK for preprocessing, and scikit-learn for Naive Bayes.

### 2.1. Libraries
- `pandas`: For data handling (DataFrame).
- `re`: For removing punctuation.
- `sklearn.feature_extraction.text.CountVectorizer`: Converts text to word count matrix.
- `sklearn.model_selection.train_test_split`: Splits data into training/testing sets.
- `sklearn.naive_bayes.MultinomialNB`: Naive Bayes classifier for count data.
- `sklearn.metrics`: For accuracy and classification report.
- `nltk.tokenize.word_tokenize`: Splits text into words.
- `nltk.corpus.stopwords`: Removes common words (e.g., “the”).
- **Fixing NLTK Error**: Download `punkt`, `punkt_tab`, `stopwords` to avoid `LookupError`.

### 2.2. Dataset
- **Sample Data**: 6 emails (3 spam, 3 ham).
  ```python
  data = {
      'email': [
          'Win a free iPhone now!!! Click here!',
          'Meeting at 10am tomorrow, please confirm.',
          'Get rich quick! Buy our course!',
          'Lunch plans this weekend? Let me know.',
          'Limited time offer! Discount viagra pills.',
          'Project deadline is next Friday.'
      ],
      'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
  }
  df = pd.DataFrame(data)
  ```
- **Role**: Provides training data to learn \( P(\text{Spam}) \), \( P(\text{Ham}) \), and word probabilities.

### 2.3. Preprocessing
- **Function**:
  ```python
  def preprocess_text(text):
      text = text.lower()
      text = re.sub(r'[^a-zA-Z\s]', '', text)
      tokens = word_tokenize(text)
      stop_words = set(stopwords.words('english'))
      tokens = [word for word in tokens if word not in stop_words]
      return ' '.join(tokens)
  df['cleaned_email'] = df['email'].apply(preprocess_text)
  ```
- **Steps**:
  - Lowercase: “Win” → “win”.
  - Remove punctuation: “!!!” → “”.
  - Tokenize: “win free” → [“win”, “free”].
  - Remove stopwords: Skip “a”, “the”.
  - Join tokens: [“win”, “free”] → “win free”.
- **Naive Bayes Role**: Creates clean word features for probability calculations.

### 2.4. Feature Extraction
- **Code**:
  ```python
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(df['cleaned_email'])
  y = df['label'].map({'spam': 1, 'ham': 0})
  ```
- **What it does**: Converts emails to a matrix (rows = emails, columns = word counts).
  - Example: “win free” → [1, 1, 0, …] (1 for “win”, 1 for “free”).
- **Naive Bayes Role**: Provides word counts to compute \( P(\text{Word} | \text{Spam}) \).

### 2.5. Training
- **Code**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  nb_classifier = MultinomialNB()
  nb_classifier.fit(X_train, y_train)
  ```
- **What it does**:
  - Splits data: 4 emails train, 2 test.
  - `MultinomialNB`: Learns:
    - \( P(\text{Spam}) \), \( P(\text{Ham}) \) from `y_train`.
    - \( P(\text{Word} | \text{Spam}) \), \( P(\text{Word} | \text{Ham}) \) from `X_train`.
- **Naive Bayes Role**: Builds the model to calculate class probabilities.

### 2.6. Evaluation
- **Code**:
  ```python
  y_pred = nb_classifier.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
  ```
- **Output** (example):
  ```
  Accuracy: 1.0
  Classification Report:
                precision    recall  f1-score   support
  Ham         1.00      1.00      1.00         1
  Spam        1.00      1.00      1.00         1
  ```
- **Metrics**:
  - **Accuracy**: Fraction of correct predictions.
  - **Precision**: Correct spam predictions / total spam predictions.
  - **Recall**: Correct spam predictions / actual spam emails.
  - **F1-score**: Harmonic mean of precision and recall.
- **Note**: High accuracy (1.0) due to small dataset; real datasets yield 85-95%.

### 2.7. Prediction
- **Code**:
  ```python
  new_email = "Win a free vacation! Click now!"
  cleaned_new_email = preprocess_text(new_email)
  new_email_vector = vectorizer.transform([cleaned_new_email])
  prediction = nb_classifier.predict(new_email_vector)
  print("New email prediction:", "Spam" if prediction[0] == 1 else "Ham")
  ```
- **What it does**: Predicts new email as spam/ham using learned probabilities.
- **Naive Bayes Role**: Computes \( P(\text{Spam} | \text{Words}) \) vs. \( P(\text{Ham} | \text{Words}) \).

## 3. Fixing NLTK LookupError
- **Issue**: `LookupError: Resource punkt_tab not found` (or `punkt`, `stopwords`).
- **Cause**: NLTK resources not downloaded or inaccessible in `~/nltk_data`.
- **Solution**:
  1. **Automated Download**:
     ```python
     nltk_data_path = os.path.expanduser('~/nltk_data')
     nltk.data.path.append(nltk_data_path)
     nltk.download('punkt', download_dir=nltk_data_path)
     nltk.download('punkt_tab', download_dir=nltk_data_path)
     nltk.download('stopwords', download_dir=nltk_data_path)
     ```
  2. **Manual Download**:
     - Download from https://www.nltk.org/nltk_data/: `punkt.zip`, `punkt_tab.zip`, `stopwords.zip`.
     - Unzip to `~/nltk_data`:
       ```bash
       mkdir -p ~/nltk_data
       unzip punkt.zip -d ~/nltk_data
       unzip punkt_tab.zip -d ~/nltk_data
       unzip stopwords.zip -d ~/nltk_data
       ```
  3. **Verify**:
     ```bash
     ls ~/nltk_data/tokenizers/punkt
     ls ~/nltk_data/tokenizers/punkt_tab
     ls ~/nltk_data/corpora/stopwords
     ```
  4. **Permissions**:
     ```bash
     chmod -R u+rw ~/nltk_data
     ```
  5. **Test Resources**:
     ```python
     from nltk.tokenize import word_tokenize
     from nltk.corpus import stopwords
     print(word_tokenize("test sentence"))
     print(stopwords.words('english')[:10])
     ```

## 4. Improvements
- **Larger Dataset**: Use UCI SMS Spam Collection:
  ```python
  df = pd.read_csv('spam.csv', encoding='latin-1')
  df = df[['v2', 'v1']].rename(columns={'v2': 'email', 'v1': 'label'})
  ```
  - URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
- **TF-IDF**:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer()
  ```
- **N-grams**:
  ```python
  vectorizer = CountVectorizer(ngram_range=(1, 2))
  ```
- **Stemming**:
  ```python
  from nltk.stem import PorterStemmer
  ps = PorterStemmer()
  def preprocess_text(text):
      text = text.lower()
      text = re.sub(r'[^a-zA-Z\s]', '', text)
      tokens = word_tokenize(text)
      stop_words = set(stopwords.words('english'))
      tokens = [ps.stem(word) for word in tokens if word not in stop_words]
      return ' '.join(tokens)
  ```

## 5. Troubleshooting
- **NLTK Error Persists**:
  - Check `nltk.data.path`:
    ```python
    import nltk
    print(nltk.data.path)
    ```
  - Reinstall NLTK:
    ```bash
    pip uninstall nltk
    pip install nltk
    ```
- **Low Accuracy**: Use larger dataset, TF-IDF, or n-grams.
- **Confusion Matrix**:
  ```python
  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
  ```