from app_store_scraper import AppStore
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

number_of_reviews = 200

# fetch app metadata from the App Store
app = AppStore(country="us", app_name="chronometer")
app.review(how_many=number_of_reviews)

# instantiate the sentiment analyzer and running total
sia = SentimentIntensityAnalyzer()
compound_running_score = 0


for review_metadata in app.reviews:
	raw_review = review_metadata["review"]
	compound_running_score += sia.polarity_scores(raw_review)["compound"] # the sentiment analyzer returns pos, neg, neutral and compound values. We use the compound for our analysis.

# compute the average compound score using our total and the number of reviews
avg_score = compound_running_score / number_of_reviews

# print the results to standard output
print("Compount sentiment score: %f" % (avg_score))

