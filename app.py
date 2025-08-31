import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

# Download dataset (only first time)
nltk.download("movie_reviews")

# Load texts and labels
fileids = movie_reviews.fileids()
X = [movie_reviews.raw(fid) for fid in fileids]
y = [movie_reviews.categories(fid)[0] for fid in fileids]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF + Logistic Regression pipeline
clf = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english",
                    ngram_range=(1, 2), min_df=2, max_df=0.95),
    LogisticRegression(max_iter=1000)
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2%}")

# Class labels
classes = clf.named_steps["logisticregression"].classes_

# Prediction function
def predict(text: str):
    if not text.strip():
        return "⚠️ Please enter text.", {"neg": 0.0, "pos": 0.0}
    probs = clf.predict_proba([text])[0]
    proba_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    label = max(proba_dict, key=proba_dict.get)
    pretty = "Positive 😀" if label == "pos" else "Negative 😡"
    return f"{pretty} ({proba_dict[label]:.2%})", proba_dict

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a review..."),
    outputs=[gr.Textbox(label="Prediction"), gr.Label(label="Confidence")],
    title="🎭 Sentiment Analysis (TF-IDF + Logistic Regression)",
    description=f"Accuracy on holdout set: {acc:.2%}. Enter a movie review or any text.",
    examples=[
        ["I absolutely loved this movie. The performances were brilliant!"],
        ["Terrible plot and wooden acting. I regret watching it."],
        ["It was fine — some good parts, some bad parts."]
    ]
)

if __name__ == "__main__":
    demo.launch()
