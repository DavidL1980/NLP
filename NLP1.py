import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#cargamos el dataset
url = 'Ruta_Dataset'
df = pd.read_json(url, lines=True)
print(df.head())

# Crear columna de sentimiento
def get_sentiment(rating):
    if rating >= 4:
        return 'positivo'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negativo'

df['sentimiento'] = df['overall'].apply(get_sentiment)
#  df.head()

print(df['sentimiento'].value_counts())
print(df[['reviewText', 'sentimiento']].head())


# Intentar descargar recursos NLTK (no mostrará salida si ya existen)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# nltk.download('punkt')
nltk.download('punkt_tab')

# Preparar stopwords y stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def preprocess_text(text):
    try: 
        if text is None:
            return ""
        if isinstance(text, float) and pd.isna(text):
            return ""
        text = str(text)

        # 1) minusculas
        text = text.lower()

        # 2) eliminar caracteres no alfanuméricos (dejamos espacios)
        text = re.sub(r'[^\w\s]', ' ', text)

        # 3) tokenizar (word_tokenize ya fue importado). Dividir en palabras
        words = word_tokenize(text)

        # 4) filtrar stopwords y tokens no útiles
        processed = []
        for w in words:
            if not w.strip(): 
                continue
            if w.isdigit():
                continue
            if w in stop_words:  
                continue
            processed.append(w)

        # 5) stemming
        processed = [stemmer.stem(w) for w in processed] 
        return " ".join(processed)

    except LookupError as e: 
        print("NLTK resource missing:", e)
        return ""
    except Exception as e: 
        print("Error en preprocess_text:", e)
        return ""

df['processed_text'] = df['reviewText'].apply(preprocess_text)
print(df[['reviewText', 'processed_text']].head())

# train y test
X = df['processed_text']
y = df['sentimiento']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)

# Modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train) 
y_pred = model.predict(X_test_tfidf)

# Evaluación
print(classification_report(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Distribución de sentimientos
df['sentimiento'].value_counts().plot(kind='bar', color=['green', 'yellow', 'red'])
plt.title('Distribución de Sentimientos en Reseñas')
plt.xlabel('Sentimiento')
plt.ylabel('Cantidad')
plt.show()


positive_text = ' '.join(df[df['sentimiento'] == 'positivo']['processed_text']) 
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text) 
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Reseñas Positivas')
plt.show()