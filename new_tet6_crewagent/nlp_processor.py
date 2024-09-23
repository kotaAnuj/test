# nlp_processor.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag, ne_chunk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def preprocess(self, text):
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        # Remove stop words and punctuation, then lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token.isalnum() and token not in self.stop_words]
        return tokens

    def extract_keywords(self, text):
        tokens = self.preprocess(text)
        # Use NLTK's FreqDist to get most common words
        freq_dist = FreqDist(tokens)
        return [word for word, _ in freq_dist.most_common(5)]

    def analyze_sentiment(self, text):
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_named_entities(self, text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        return [(entity.label(), ' '.join(word for word, tag in entity.leaves()))
                for entity in named_entities if isinstance(entity, nltk.Tree)]

    def analyze_text(self, text):
        sentences = sent_tokenize(text)
        keywords = self.extract_keywords(text)
        sentiment = self.analyze_sentiment(text)
        named_entities = self.extract_named_entities(text)
        
        return {
            'sentences': sentences,
            'keywords': keywords,
            'sentiment': sentiment,
            'named_entities': named_entities
        }