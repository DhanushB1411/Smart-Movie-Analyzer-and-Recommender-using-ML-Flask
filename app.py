import os
import pickle
import numpy as np
import google.generativeai as genai
import secrets

# Generate a secure random key
secret_key = secrets.token_hex(16)

from flask import Flask,render_template,request,redirect,url_for,session,flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.secret_key = secret_key


app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Movies(db.Model):
    movieid = db.Column(db.Integer,primary_key = True)
    moivename = db.Column(db.String(100), nullable=False)
    review = db.Column(db.String(100), nullable=False)
    emotion = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())

    def __repr__(self):
        return f'<Movie {self.moivename}>'


model = pickle.load(open('model.pkl','rb'))

# Load the vectorizer along with the model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

from nltk.tokenize import RegexpTokenizer
#for tokexizing the data into
# "my name is"  => ["my","name","is"]
from nltk.stem.porter import PorterStemmer
#cleaning the data like "liking " -> "like"
from nltk.corpus import  stopwords
# to remove the unwanted data like the is
import nltk
nltk.download('stopwords')
# Downloading the stopwords
#tokenizer with spaceblank
tokenizer = RegexpTokenizer(r"\w+")

en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
def getCleanedText(text):
  text = text.lower()
  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text



@app.route("/")
def hello_world():
    reviews = Movies.query.all()
    return render_template('index.html',Reviews = reviews)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        review = request.form['review']
        cleaned_review = getCleanedText(review)
        rev = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(rev)
        pred = 'Positive' if prediction[0] == 1 else 'Negative'
        prediction_text = 'POSITIVE RESPONSE' if prediction[0] == 1 else 'NEGATIVE RESPONSE'
        
        new_review = Movies(moivename='fight Club', review=review, emotion=pred)
        db.session.add(new_review)
        db.session.commit()
        
        reviews = Movies.query.all()
        return render_template('movie.html', prediction_text=prediction_text, Reviews=reviews)
    except Exception as e:
        return str(e)



@app.route("/film")
def home():
        reviews = Movies.query.filter_by(moivename = 'Fight Club')
        return render_template("movie.html",Reviews = reviews)


@app.route("/batman",methods = ['POST','GET'])
def batman():
    #movie_name = request.args.get('movie', 'Batman')
    movie_name = request.args.get('movie')
    img_name = request.args.get('img')
    if request.method == 'POST':
        try:
            review = request.form['review']
            movie_name = request.form['movie']
            cleaned_review = getCleanedText(review)
            rev = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(rev)

            pred = 'Positive' if prediction[0] == 1 else 'Negative'
            prediction_text = 'POSITIVE RESPONSE' if prediction[0] == 1 else 'NEGATIVE RESPONSE'

            new_review = Movies(moivename=movie_name, review=review, emotion=pred)
            db.session.add(new_review)
            db.session.commit()

            reviews = Movies.query.filter_by(moivename = movie_name)
            positive_count = Movies.query.filter_by(moivename = movie_name,emotion='Positive').count()
            negative_count = Movies.query.filter_by(moivename = movie_name,emotion='Negative').count()
            return render_template('check.html', prediction_text=prediction_text, Reviews=reviews,positive_count=positive_count, negative_count=negative_count,img_name = img_name)
        except Exception as e:
            return str(e)
    else:
        reviews = Movies.query.filter_by(moivename = movie_name)
        positive_count = Movies.query.filter_by(moivename = movie_name,emotion='Positive').count()
        negative_count = Movies.query.filter_by(moivename = movie_name,emotion='Negative').count()
        return render_template('check.html',Reviews=reviews,movie_name = movie_name,positive_count=positive_count, negative_count=negative_count,img_name = 'thebatman.png')



@app.route("/count_reviews")
def count_reviews():
    positive_count = Movies.query.filter_by(moivename = 'fight Club',emotion='Positive').count()
    negative_count = Movies.query.filter_by(moivename = 'fight Club',emotion='Negative').count()
    return render_template('count.html', positive_count=positive_count, negative_count=negative_count)


@app.route("/suggest_movie")
def suggest_movie():
    # Subquery to count positive and negative reviews for each movie
    positive_reviews = db.session.query(
        Movies.moivename,
        db.func.count(Movies.emotion).label('positive_count')
    ).filter_by(emotion='Positive').group_by(Movies.moivename).subquery()

    negative_reviews = db.session.query(
        Movies.moivename,
        db.func.count(Movies.emotion).label('negative_count')
    ).filter_by(emotion='Negative').group_by(Movies.moivename).subquery()

    # Main query to calculate the difference and rank movies
    ranked_movies = db.session.query(
        positive_reviews.c.moivename,
        (positive_reviews.c.positive_count - db.func.coalesce(negative_reviews.c.negative_count, 0)).label('review_score')
    ).outerjoin(
        negative_reviews, positive_reviews.c.moivename == negative_reviews.c.moivename
    ).order_by(db.desc('review_score')).all()

    return render_template('suggestions.html', movies=ranked_movies)
    


@app.route("/login_page", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Hardcoded admin credentials for simplicity (you can move this to a database)
        if username == "admin" and password == "password123":
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for("admin_login"))

    return render_template("admin_login.html")


@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    reviews = Movies.query.all()
    return render_template("admin_dashboard.html", Reviews=reviews)


@app.route("/admin/delete_review/<int:id>", methods=["POST"])
def delete_review(id):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    
    review = Movies.query.get(id)
    db.session.delete(review)
    db.session.commit()
    flash("Review deleted successfully!")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/add_review", methods=["POST"])
def add_review():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    try:
        movie_name = request.form['movie_name']
        review = request.form['review']
        emotion = request.form['emotion']

        new_review = Movies(moivename=movie_name, review=review, emotion=emotion)
        db.session.add(new_review)
        db.session.commit()
        flash("Review added successfully!")
    except Exception as e:
        flash(f"Error adding review: {str(e)}")
    
    return redirect(url_for("admin_dashboard"))

if __name__ == "__main__":
    app.run(debug=True)

