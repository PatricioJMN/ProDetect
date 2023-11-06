import os
import json
import librosa
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to load audio data (you need to implement this based on your dataset).
def load_audio_data(audio_path):
    try:
        # Debugging: Print the audio_path to check if it's correct.
        # print("Loading audio from:", audio_path)

        # Implement audio loading from your dataset.
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        # Debugging: Print success message.
        # print("Audio loaded successfully.")

        return audio_data, sample_rate
    except Exception as e:
        # Debugging: Print any errors that occur during loading.
        # print("Error loading audio:", str(e))
        return None, None

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the learning curve.

    Parameters:
    - estimator: Your XGBoost classifier.
    - title: Title for the chart.
    - X: Feature matrix.
    - y: Target vector.
    - ylim: Tuple (min, max) to define the y-axis limits.
    - cv: Cross-validation splitting strategy.
    - n_jobs: Number of CPU cores to use for parallel computation.
    - train_sizes: Array of training set sizes.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Load the pronunciation scores from scores.json.
scores_path = os.path.join('dataset/resource/scores.json')
with open(scores_path, 'r') as scores_file:
    pronunciation_scores = json.load(scores_file)

# Define variables to store features and labels of training and testing.
X_train = []  # Features (MFCCs)
y_train = []  # Labels (1 for mispronunciation, 0 for correct pronunciation)
X_test = []
y_test = []

# Define a dictionary to store mappings between utterance IDs and audio paths.
utt2audio_path = {}

# Read the wav.scp file of the train dataset and populate the utt2audio_path dictionary.
with open(os.path.join('dataset/train/wav.scp'), 'r') as wav_scp_file:
    for line in wav_scp_file:
        utt_id, audio_path = line.strip().split(maxsplit=1)
        utt2audio_path[utt_id] = audio_path

# Loop through your dataset to extract features and labels.
for utt_id, pronunciation_score in pronunciation_scores.items():
    # Extract relevant information from pronunciation_score.
    text = pronunciation_score["text"]
    accuracy = pronunciation_score["accuracy"]
    total = pronunciation_score["total"]
    audio_path = utt2audio_path.get(utt_id)
    audio_path = f"dataset/{audio_path}"
    # print(audio_path)
    if audio_path:
        audio_data, sample_rate = load_audio_data(audio_path)
        if audio_data is not None:
            # Extract MFCC features from audio.
            mfcc_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=2600).T, axis=0)
            # Append MFCC features and label to X and y.
            X_train.append(mfcc_features)
            # Define a threshold or criteria to determine if it's a mispronunciation.
            # For example, if accuracy < threshold, consider it a mispronunciation.
            threshold = 7  # You should adjust this threshold based on your dataset.
            label = 1 if total < threshold else 0
            y_train.append(label)

# Convert X and y to numpy arrays.
X_train = np.array(X_train)
y_train = np.array(y_train)

# Split the dataset into training and testing sets.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.2, random_state=42)

# Initialize and train an XGBoost classifier.
clf = XGBClassifier(learning_rate=0.5, random_state=42, n_estimators=10, reg_alpha=1, reg_lambda=1)
clf.fit(X_train, y_train)

# Read the wav.scp file of the test dataset and populate the utt2audio_path dictionary.
with open(os.path.join('dataset/test/wav.scp'), 'r') as wav_scp_file:
    for line in wav_scp_file:
        utt_id, audio_path = line.strip().split(maxsplit=1)
        utt2audio_path[utt_id] = audio_path

# Loop through your dataset to extract features and labels.
for utt_id, pronunciation_score in pronunciation_scores.items():
    # Extract relevant information from pronunciation_score.
    text = pronunciation_score["text"]
    accuracy = pronunciation_score["accuracy"]
    total = pronunciation_score["total"]
    audio_path = utt2audio_path.get(utt_id)
    audio_path = f"dataset/{audio_path}"
    # print(audio_path)
    if audio_path:
        audio_data, sample_rate = load_audio_data(audio_path)
        if audio_data is not None:
            # Extract MFCC features from audio.
            mfcc_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=2600).T, axis=0)
            # Append MFCC features and label to X and y.
            X_test.append(mfcc_features)
            # Define a threshold or criteria to determine if it's a mispronunciation.
            # For example, if accuracy < threshold, consider it a mispronunciation.
            threshold = 7  # You should adjust this threshold based on your dataset.
            label = 1 if total < threshold else 0
            y_test.append(label)

# Convert X and y to numpy arrays.
X_test = np.array(X_test)
y_test = np.array(y_test)

# Make predictions on the test set.
y_pred = clf.predict(X_test)

# Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model to a file.
model_filename = 'model/prodetect_copy.pkl'
joblib.dump(clf, model_filename)

# With the already trained your XGBoost model and have X and y data
title = "Learning Curve (XGBoost)"
cv = 4  # Number of cross-validation folds
n_jobs = -1  # Use all available CPU cores for parallel computation

plot_learning_curve(clf, title, X_test, y_test, cv=cv, n_jobs=n_jobs)
plt.show()
