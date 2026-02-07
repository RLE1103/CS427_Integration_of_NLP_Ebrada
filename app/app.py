from flask import Flask, render_template, request
import joblib
import os

# Get the directory where this app.py file is located
app_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of app directory)
project_root = os.path.dirname(app_dir)

# Create Flask app with custom template and static folders
app = Flask(__name__, 
            template_folder=os.path.join(project_root, "templates"),
            static_folder=os.path.join(project_root, "static"))

# Load model and vectorizer from the app directory
model = joblib.load(os.path.join(app_dir, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(app_dir, "tfidf_vectorizer.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        vector = vectorizer.transform([text])
        sentiment = model.predict(vector)[0]
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
