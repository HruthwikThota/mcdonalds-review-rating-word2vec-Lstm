# McDonald's Customer Review Rating Prediction Using Word2Vec + LSTM

## 1. Project Overview

This project focuses on predicting customer ratings (1 to 5 stars) from McDonald's customer reviews using Natural Language Processing (NLP) and Deep Learning.

The project explores three different approaches using the same dataset:

1. TF-IDF + Artificial Neural Network (ANN)
2. Word2Vec + Artificial Neural Network (ANN)
3. Word2Vec + Long Short-Term Memory (LSTM)

The main objective was to understand how different text representation techniques and neural network architectures affect sentiment classification performance.

The final experiment uses Word2Vec embeddings with an LSTM network. Optuna was used for hyperparameter optimization to improve model performance.

**Final tuned model test accuracy: 65.92%.**

---

## 2. Project Objectives

- Perform text preprocessing and cleaning on customer reviews.
- Convert textual reviews into numerical representations.
- Understand the differences between TF-IDF and Word2Vec.
- Build ANN and LSTM classification models.
- Compare different NLP and Deep Learning approaches.
- Apply hyperparameter tuning using Optuna.
- Evaluate models using classification metrics and confusion matrices.
- Deploy the trained model using Streamlit.
- Understand how word embeddings and sequential learning work in NLP.

---

## 3. Dataset Information

**Dataset:** McDonald's Customer Reviews

**Original dataset size:** 33,396 rows and 10 columns.

**Columns:**

| Column | Description |
|---|---|
| reviewer_id | Unique reviewer identifier |
| store_name | Restaurant name |
| category | Restaurant category |
| store_address | Restaurant address |
| latitude | Store latitude |
| longitude | Store longitude |
| rating_count | Number of ratings |
| review_time | Review submission time |
| review | Customer review text |
| rating | Customer rating |

**Target variable:** `rating`

The original ratings were available in text format:

- 1 star
- 2 stars
- 3 stars
- 4 stars
- 5 stars

These ratings were converted into numerical labels.

| Original Rating | Encoded Value |
|---|---:|
| 1 star | 0 |
| 2 stars | 1 |
| 3 stars | 2 |
| 4 stars | 3 |
| 5 stars | 4 |

Fifty rows with missing reviews were removed, leaving 33,346 rows before further text cleaning.

---

## 4. Common Data Preprocessing

The same dataset, cleaning approach, target labels and train-test split were used across the experiments to support a fair comparison.

### Step 1: Remove Missing Reviews

Removed rows where the review column contained missing values.

### Step 2: Rating Encoding

Converted the five rating categories into numerical values from 0 to 4.

### Step 3: Text Cleaning

Applied the following operations:

- Converted text into lowercase.
- Removed URLs.
- Removed HTML tags.
- Removed numbers and special characters.
- Preserved apostrophes and negation words.
- Removed extra spaces.

Example:

Original review:

"THE FOOD WAS GOOD!!! Visit www.example.com"

Cleaned review:

"the food was good"

### Step 4: Remove Empty Reviews

Removed reviews that became empty after text cleaning.

### Step 5: Separate Features and Target

- X: Cleaned customer reviews.
- y: Numerical rating labels.

### Step 6: Train-Test Split

Used Scikit-learn's `train_test_split()`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Dataset split:

| Dataset | Number of Reviews |
|---|---:|
| Training | 26,676 |
| Testing | 6,670 |
| Total | 33,346 |

`stratify=y` maintained the rating distribution across training and testing datasets.

`random_state=42` ensured reproducibility.

---

# 5. Experiment 1: TF-IDF + ANN

### Approach

The first experiment used TF-IDF (Term Frequency-Inverse Document Frequency) to convert customer reviews into numerical feature vectors.

These vectors were passed into an Artificial Neural Network for five-class rating classification.

### TF-IDF

TF-IDF assigns numerical importance to words based on their frequency in a document and across the entire document collection.

It also captures useful word combinations through unigrams and bigrams.

### Model Architecture

- Input: TF-IDF feature vectors.
- Hidden layers: ANN Dense layers.
- Output: Five neurons with Softmax activation.

### Results

| Metric | Result |
|---|---:|
| Test Accuracy | 68.40% |
| Macro F1-score | 0.62 |
| Weighted F1-score | 0.67 |

### Key Learning

TF-IDF performed well because it captured useful individual words and word combinations.

Bigrams helped represent phrases such as "not good" and "very bad".

This experiment became the baseline for the later comparisons.

---

# 6. Experiment 2: Word2Vec + ANN

### Approach

The second experiment replaced TF-IDF with Word2Vec embeddings.

Word2Vec learns dense numerical representations of words based on their surrounding context.

### Word2Vec Configuration

| Parameter | Value |
|---|---:|
| Algorithm | Skip-gram |
| Vector Size | 100 |
| Window Size | 5 |
| Minimum Count | 1 |
| Epochs | 10 |
| sg | 1 |

Word2Vec was trained only on the training reviews to avoid data leakage.

### Review Representation

Each review contained multiple words.

To convert an entire review into one vector, the average of its word embeddings was calculated.

This produced a 100-dimensional vector for every review.

### ANN Architecture

The averaged Word2Vec vectors were passed into an Artificial Neural Network for five-class classification.

### Results

| Metric | Result |
|---|---:|
| Test Accuracy | 64.83% |
| Test Loss | 0.8963 |
| Macro F1-score | 0.52 |
| Weighted F1-score | 0.61 |

### Problems Identified

The Word2Vec + ANN model performed worse than TF-IDF + ANN.

Important observations:

- Averaging word embeddings loses word order.
- The model receives one vector for the entire review.
- Some contextual information is lost during averaging.
- The model struggled to distinguish middle ratings, particularly 2-star and 3-star reviews.

### Key Learning

Word2Vec learns meaningful word representations, but averaging embeddings does not preserve the sequence of words.

This motivated the next experiment using LSTM.

---

# 7. Experiment 3: Word2Vec + LSTM

### Objective

The third experiment combined Word2Vec embeddings with an LSTM network.

Unlike the previous ANN experiment, this approach preserves the sequence of words in each review.

### Why LSTM?

LSTM stands for Long Short-Term Memory.

It is a type of Recurrent Neural Network (RNN) designed to learn relationships across sequential data.

In NLP, LSTM processes words in sequence and learns contextual patterns across a review.

This approach was selected to address the loss of word order caused by averaging Word2Vec embeddings.

### Step 1: Tokenization

Used Keras Tokenizer to convert words into numerical IDs.

The tokenizer was fitted only on the training reviews.

Vocabulary size:

12,665, including the padding index.

### Step 2: Sequence Conversion

Each review was converted into a sequence of integer IDs.

Example:

"the food was good"

[12, 45, 23, 89]

The actual IDs depend on the learned vocabulary.

### Step 3: Padding

LSTM requires sequences of equal length.

The maximum sequence length was set to 80.

```python
maxlen = 80

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=maxlen,
    padding='post',
    truncating='post'
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=maxlen,
    padding='post',
    truncating='post'
)
```

- `padding='post'`: Adds zeros at the end of shorter sequences.
- `truncating='post'`: Removes words from the end of longer sequences.

### Step 4: Train Word2Vec

Word2Vec was trained using the training reviews only.

```python
word2vec_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=10
)
```

### Step 5: Create Embedding Matrix

Created an embedding matrix using the Word2Vec vocabulary.

Embedding matrix shape:

(12665, 100)

- 12,665 rows represent vocabulary entries, including the padding index.
- 100 columns represent the dimensions of each word vector.

This matrix was used to initialize the Keras Embedding layer.

### Step 6: LSTM Model Architecture

The model contained three main layers:

| Layer | Configuration |
|---|---|
| Embedding | 100-dimensional Word2Vec embeddings |
| LSTM | 64 units |
| Dense | 5 neurons, Softmax activation |

The Embedding layer was initialized with Word2Vec weights and set to `trainable=True`.

This allowed the embeddings to be fine-tuned during supervised training.

### Model Compilation

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

Why these settings?

- Sparse categorical crossentropy: Suitable for five-class classification with integer labels.
- Adam: Optimizer for updating model weights.
- Accuracy: Measures correct predictions.

### Initial Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 15 |
| Batch Size | 64 |
| Validation Split | 20% |
| LSTM Units | 64 |
| Dropout | 0.2 |
| Recurrent Dropout | 0.2 |

### Initial Model Results

| Metric | Result |
|---|---:|
| Test Accuracy | 65.23% |
| Test Loss | 0.9926 |
| Macro F1-score | 0.59 |
| Weighted F1-score | 0.65 |

### Classification Report

| Rating | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 1 Star | 0.77 | 0.83 | 0.80 |
| 2 Stars | 0.37 | 0.37 | 0.37 |
| 3 Stars | 0.49 | 0.52 | 0.50 |
| 4 Stars | 0.54 | 0.49 | 0.52 |
| 5 Stars | 0.76 | 0.73 | 0.74 |

### Initial Training Observations

- Training accuracy reached 78.10%.
- Best validation accuracy was 67.86%.
- Validation loss reached its minimum of 0.8987 at Epoch 8.
- Training loss continued decreasing while validation loss increased.

This indicated overfitting.

---

# 8. Hyperparameter Tuning Using Optuna

### Why Optuna?

The initial LSTM model showed signs of overfitting.

Optuna was introduced to search for better hyperparameter combinations.

### Hyperparameters Tuned

| Hyperparameter | Search Range |
|---|---|
| LSTM Units | 32, 64, 128 |
| Dropout | 0.1 to 0.5 |
| Recurrent Dropout | Initially 0.0 to 0.3 |
| Learning Rate | 0.0001 to 0.01 |
| Batch Size | 32, 64, 128 |

### Optimization Objective

The objective function trained a new LSTM model for each trial.

Optuna maximized validation accuracy.

Early Stopping was used to stop training when validation loss stopped improving.

### First Tuning Attempt

The first CPU-based study used 20 trials.

The search was slow because each trial trained a separate LSTM model.

The initial six completed trials produced a best validation accuracy of approximately 67.90%.

### GPU Optimization

The runtime was changed to T4 GPU.

However, training remained slow because non-zero `recurrent_dropout` prevented the standard fast cuDNN LSTM implementation from being used.

### Optimization Improvement

The objective function was modified:

```python
recurrent_dropout = 0.0
```

This allowed TensorFlow to use its faster GPU-compatible LSTM implementation when the other cuDNN requirements were met.

A fresh Optuna study was created with 10 trials.

### Best Optuna Results

| Hyperparameter | Best Value |
|---|---:|
| LSTM Units | 64 |
| Dropout Rate | 0.16385 |
| Recurrent Dropout | 0.0 |
| Learning Rate | 0.002014 |
| Batch Size | 128 |

**Best validation accuracy: 68.38%.**

The 10-trial optimization completed in approximately five minutes on the T4 GPU.

---

# 9. Final Tuned LSTM Model

The final model was rebuilt using the best Optuna hyperparameters.

### Final Architecture

| Layer | Configuration |
|---|---|
| Input | 80 tokens |
| Embedding | 100 dimensions |
| LSTM | 64 units |
| Dropout | 0.16385 |
| Recurrent Dropout | 0.0 |
| Output | Dense(5), Softmax |

### Early Stopping

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

Early Stopping monitored validation loss and restored the weights from the best validation-loss epoch.

### Final Training Observations

- Training stopped after Epoch 10.
- Best validation accuracy during final training: 67.28% at Epoch 7.
- Lowest validation loss: 0.8897 at Epoch 7.
- Training accuracy at Epoch 10: 79.24%.
- Validation accuracy at Epoch 10: 65.09%.

### Final Test Results

| Metric | Result |
|---|---:|
| Test Accuracy | **65.92%** |
| Test Loss | **0.9166** |
| Macro F1-score | 0.58 |
| Weighted F1-score | 0.65 |

### Final Classification Report

| Rating | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 1 Star | 0.76 | 0.84 | 0.80 |
| 2 Stars | 0.38 | 0.30 | 0.33 |
| 3 Stars | 0.46 | 0.53 | 0.49 |
| 4 Stars | 0.64 | 0.40 | 0.49 |
| 5 Stars | 0.73 | 0.81 | 0.77 |

### Final Confusion Matrix

| Actual / Predicted | 1 Star | 2 Stars | 3 Stars | 4 Stars | 5 Stars |
|---|---:|---:|---:|---:|---:|
| 1 Star | 1582 | 142 | 114 | 15 | 33 |
| 2 Stars | 284 | 183 | 112 | 13 | 25 |
| 3 Stars | 124 | 107 | 509 | 80 | 142 |
| 4 Stars | 52 | 28 | 200 | 458 | 418 |
| 5 Stars | 49 | 27 | 161 | 147 | 1665 |

### Key Observations

- The model performs better on extreme ratings, particularly 1-star and 5-star reviews.
- 2-star reviews remain difficult to identify.
- Many 2-star reviews are classified as 1-star.
- Many 4-star reviews are classified as 5-star.
- Middle ratings show considerable overlap in their language and sentiment.

Hyperparameter tuning improved test accuracy from 65.23% to 65.92%, an improvement of 0.69 percentage points.

Test loss also decreased from 0.9926 to 0.9166.

However, not every rating category improved. Macro F1 decreased slightly from 0.59 to 0.58.

---

# 10. Comparison of All Experiments

All experiments used the same dataset and 80:20 train-test split.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + ANN | 68.40% | 0.62 | 0.67 |
| Word2Vec + ANN | 64.83% | 0.52 | 0.61 |
| Word2Vec + LSTM | 65.23% | 0.59 | 0.65 |
| Tuned Word2Vec + LSTM | 65.92% | 0.58 | 0.65 |

### Comparison and Learning

**TF-IDF + ANN**

- Achieved the highest test accuracy.
- TF-IDF captured useful word frequencies and bigram features.
- Demonstrated that a simpler text representation performed strongly on this dataset.

**Word2Vec + ANN**

- Learned dense word representations.
- Averaging word vectors removed word order.
- Performance decreased compared with TF-IDF + ANN.

**Word2Vec + LSTM**

- Used sequential word representations.
- LSTM learned patterns across the review sequence.
- Improved upon Word2Vec + ANN in accuracy and macro F1.

**Tuned Word2Vec + LSTM**

- Optuna identified a better hyperparameter configuration.
- Test accuracy improved by 0.69 percentage points.
- Test loss decreased.
- TF-IDF + ANN remained the best-performing model overall.

### Overall Learning

The experiments demonstrated that text representation and model architecture both influence classification performance.

A more complex architecture does not automatically produce higher accuracy.

---

# 11. Challenges Faced and Solutions

| Challenge | Solution |
|---|---|
| Missing review values | Removed rows with missing reviews |
| Text contained URLs and special characters | Applied regular-expression-based cleaning |
| Ratings were stored as text | Converted ratings into numerical labels |
| Neural networks require numerical input | Applied TF-IDF or Word2Vec |
| LSTM requires equal-length sequences | Applied padding and truncation |
| Word2Vec vocabulary needed to align with tokenizer | Created an embedding matrix using tokenizer word indices |
| Risk of data leakage | Trained tokenizer and Word2Vec using training data only |
| Word2Vec averaging lost word order | Introduced LSTM to process sequential inputs |
| Initial LSTM showed overfitting | Applied Dropout and Early Stopping |
| Optuna tuning took too long | Switched to T4 GPU and set recurrent dropout to zero |
| Different hyperparameters produced different results | Used Optuna to search for the best configuration |
| Model needed to work on unseen reviews | Saved the trained model and tokenizer for inference |

---

# 12. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Regular Expressions | Text cleaning |
| Scikit-learn | Train-test split and evaluation |
| Gensim | Word2Vec training |
| TensorFlow / Keras | ANN and LSTM development |
| Optuna | Hyperparameter optimization |
| Matplotlib | Training graphs |
| Seaborn | Confusion matrix visualization |
| Streamlit | Web application deployment |
| GitHub | Version control and project hosting |

---

# 13. Saved Model Files

The following files were saved in Google Drive.

| File | Purpose |
|---|---|
| `final_word2vec_lstm_model.keras` | Final tuned LSTM model |
| `word2vec_model.model` | Trained Word2Vec model |
| `tokenizer.pkl` | Fitted Keras tokenizer |
| `best_lstm_hyperparameters.json` | Best Optuna hyperparameters and model configuration |
| `optuna_trial_results.csv` | Results of all 10 Optuna trials |

The Word2Vec vectors were already stored in the trained Keras Embedding layer. Therefore, the Streamlit prediction application does not need to load the separate Gensim model.

---

# 14. Streamlit Application

The final tuned model was prepared for deployment using Streamlit.

### Application Workflow

1. User enters a customer review.
2. Application cleans the review.
3. Saved tokenizer converts words into numerical sequences.
4. Padding converts the sequence to 80 tokens.
5. Trained LSTM model processes the sequence.
6. Softmax produces probabilities for five rating classes.
7. The application displays the predicted rating.

### Prediction Mapping

| Model Output | Displayed Rating |
|---|---|
| 0 | 1 Star |
| 1 | 2 Stars |
| 2 | 3 Stars |
| 3 | 4 Stars |
| 4 | 5 Stars |

The application also displays the highest Softmax score as a confidence score. This score is not a calibrated measure of prediction reliability.

### Required Deployment Files

```text
mcdonalds-review-rating-word2vec-lstm/
│
├── app.py
├── requirements.txt
├── final_word2vec_lstm_model.keras
├── tokenizer.pkl
├── best_lstm_hyperparameters.json
├── word2vec_model.model
└── README.md
```

---

# 15. Future Improvements

Possible future improvements include:

- Experiment with Bidirectional LSTM.
- Try GRU architecture.
- Experiment with different maximum sequence lengths.
- Explore class weighting to address weaker performance on minority rating classes.
- Experiment with pretrained Word2Vec or GloVe embeddings.
- Try BERT-based sentiment classification.
- Perform systematic error analysis on misclassified reviews.
- Evaluate macro F1 and per-class recall as additional optimization objectives.
- Apply probability calibration if confidence estimates are needed.

The next planned experiment is BERT-based classification using the same dataset and test split.

---

