import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os


#cargamos el dataset
url = 'Ruta_del_dataset'  # Reemplaza con la ruta correcta
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
        # Forzar a string (por seguridad)
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
            # opcional: ignorar números puros
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

# Aplicar la función (es segura ante NaN)
df['processed_text'] = df['reviewText'].apply(preprocess_text)

print(df[['reviewText', 'processed_text']].head())

# Preparar datos para BERT (usa etiquetas numéricas: 0=negativo, 1=neutro, 2=positivo)
label_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
df['label'] = df['sentimiento'].map(label_map)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt") for text in df['reviewText']]
        self.labels = [label for label in df['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids': self.texts[idx]['input_ids'][0], 'attention_mask': self.texts[idx]['attention_mask'][0], 'labels': torch.tensor(self.labels[idx])}

train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)

model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluación
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
print("F1-Score (macro):", f1_score(test_df['label'], y_pred, average='macro'))

# PASO 1: LIMPIEZA DE DATOS (mismo)
print("Dataset original:", len(df))
df_clean = df.dropna(subset=['reviewText', 'overall'])
df_clean = df_clean[df_clean['reviewText'].str.len() > 10]
print("Dataset limpio:", len(df_clean))

label_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
df_clean['label'] = df_clean['sentimiento'].map(label_map)
df_clean = df_clean.dropna(subset=['label'])
print("Dataset final para BERT:", len(df_clean))
print("Distribución de labels:", df_clean['label'].value_counts())

""" 
Define un mapeo de etiquetas textuales a enteros y crea la columna label con esos números. Filtra valores desconocidos a NaN.
"""

train_df, test_df = train_test_split(
    df_clean, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_clean['label']
)

"""
divide los datos en entrenamiento y prueba, manteniendo la proporción de clases con stratify.
"""

# PASO 2: TOKENIZER Y DATASET (mismo)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset): 
    def __init__(self, dataframe, tokenizer, max_length=128):  
        self.tokenizer = tokenizer 
        self.data = []  
        
        for idx, row in dataframe.iterrows(): 
            text = str(row['reviewText'])
            encoding = tokenizer(
                text,
                padding='max_length', 
                truncation=True, 
                max_length=max_length,
                return_tensors='pt'  
            )

            self.data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(row['label'], dtype=torch.long)
            })

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):  
        return self.data[idx]

train_dataset = SentimentDataset(train_df, tokenizer)
test_dataset = SentimentDataset(test_df, tokenizer)

print(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
print(f"Dataset de test: {len(test_dataset)} muestras")


# PASO 3: MODELO BERT
model_bert = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=3
)

# PASO 4: CONFIGURACIÓN CORREGIDA PARA TRANSFORMERS 4.57.0
training_args = TrainingArguments(
    output_dir='./results_bert', 
    num_train_epochs=2,  
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4, 
    warmup_steps=100,  
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",        
    save_strategy="steps",          
    eval_steps=50,                   
    save_steps=100,                 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to=None,                 
    dataloader_num_workers=0,       
    remove_unused_columns=False,    
)


# PASO 5: MÉTRICAS PERSONALIZADAS
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1
    }

# PASO 6: TRAINER
trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# PASO 7: ENTRENAMIENTO
print("Iniciando entrenamiento de BERT...")
trainer.train()
print("Entrenamiento completado!")

# PASO 8: EVALUACIÓN
print("Evaluando modelo...")
eval_results = trainer.evaluate()
print("Resultados de evaluación:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Reporte detallado
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = np.array([item['labels'].item() for item in test_dataset])

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=['Negativo', 'Neutro', 'Positivo']))

f1_final = f1_score(y_true, y_pred, average='macro')
print(f"\nF1-Score final (macro): {f1_final:.4f}")

# PASO 9: FUNCIÓN DE PREDICCIÓN
def predict_sentiment(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    
    sentiment_map = {0: 'negativo', 1: 'neutro', 2: 'positivo'}
    confidence = probs.max().item()
    
    return sentiment_map[predicted_class], confidence, probs[0].numpy()

# Demo interactivo
demo_reviews = [
    "Love this lipstick! Perfect color and lasts all day.",
    "Average quality, nothing special about it.",
    "Terrible product, broke after one use. Waste of money!"
]

print("\n=== DEMO DE PREDICCIONES ===")
for i, review in enumerate(demo_reviews, 1):
    sentiment, confidence, probabilities = predict_sentiment(review, model_bert, tokenizer)
    print(f"{i}. '{review[:50]}...'")
    print(f"   Sentimiento: {sentiment}")
    print(f"   Confianza: {confidence:.3f}")
    print(f"   Probabilidades: Neg: {probabilities[0]:.3f}, Neu: {probabilities[1]:.3f}, Pos: {probabilities[2]:.3f}")
    print()

# Guardar modelo
model_bert.save_pretrained('./bert_sentiment_model')
tokenizer.save_pretrained('./bert_sentiment_model')
print("✅ Modelo guardado en './bert_sentiment_model'")

# Guardar métricas para portafolio
metrics = {
    'f1_macro': f1_final,
    'accuracy': accuracy_score(y_true, y_pred),
    'dataset_size': len(df_clean),
    'train_size': len(train_dataset),
    'test_size': len(test_dataset)
}

import json
with open('bert_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✅ Métricas guardadas en 'bert_metrics.json'")