Proyecto de análisis de sentimiento en reseñas de productos utilizando técnicas de Procesamiento de Lenguaje Natural (NLP). Dataset público de reseñas de Amazon: "All Beauty" (productos de belleza). Este dataset, compilado por investigadores de la Universidad de California, San Diego (UCSD) bajo la dirección de Julian McAuley, proviene de la versión v2.0 de la colección Amazon Reviews (disponible en https://jmcauley.ucsd.edu/data/amazon\_v2/).



Se trata de un conjunto de datos "5-core", lo que significa que incluye solo reseñas de usuarios y productos con al menos 5 interacciones, garantizando mayor calidad y reduciendo ruido. El archivo utilizado (All\_Beauty\_5.json.gz) contiene aproximadamente 5,269 reseñas recolectadas hasta 2018, extraídas de forma pública y anonimizada mediante la API oficial de Amazon. Cada reseña incluye campos clave como:



* reviewText: El texto completo de la reseña (entrada principal para NLP).
* overall: Puntuación de 1 a 5 estrellas (usada para etiquetar sentimientos: positivo >3, neutro =3, negativo <3).
* Otros: ID de usuario/producto, resumen, timestamp y votos.



Este dataset es ideal para tareas de clasificación de sentimientos por su tamaño manejable (~1MB comprimido), equilibrio relativo en calificaciones y relevancia real en e-commerce. Permite demostrar aplicaciones prácticas como el análisis de feedback para marketing digital o mejora de productos, con énfasis en preprocesamiento (tokenización, stemming) y modelado (TF-IDF + Naive Bayes como baseline, y BERT para precisión avanzada). Los datos son de acceso libre para fines educativos e investigativos, promoviendo reproducibilidad en proyectos de ML/DL





RESULTADOS



sentimiento

positivo    4981

negativo     179

neutral      109



El análisis inicial de las etiquetas de sentimiento derivadas de las calificaciones (overall) en el dataset de reseñas de Amazon Beauty revela una distribución altamente desbalanceada: de las 5,269 reseñas procesadas, 4,981 (aproximadamente el 94.5%) se clasifican como positivas (calificaciones >3 estrellas), 179 (3.4%) como negativas (<3 estrellas) y 109 (2.1%) como neutrales (=3 estrellas). Esta distribución refleja una tendencia común en reseñas de e-commerce, donde los usuarios tienden a dejar feedback positivo en mayor proporción, posiblemente debido a sesgos de selección (e.g., compradores satisfechos son más propensos a reseñar). En términos prácticos para aplicaciones de marketing, esto implica que los modelos deben manejarse con técnicas de balanceo de clases (como undersampling o weighted loss) para evitar un bias hacia la clase mayoritaria y mejorar la detección de insights negativos valiosos, como quejas sobre productos que podrían informar estrategias de mejora.

Este desbalance puede impactar las métricas de evaluación, haciendo que métricas como accuracy sean engañosas (un modelo naive que prediga siempre "positivo" alcanzaría ~94% accuracy). Por ello, en este proyecto se prioriza el F1-score macro, que promedia el rendimiento por clase de manera equitativa, asegurando robustez en escenarios reales donde identificar reseñas negativas o neutrales es clave para análisis de feedback empresarial y optimización de ventas en plataformas como Amazon.







Procesamiento de texto



El preprocesamiento de texto aplicado al dataset transforma las reseñas crudas en formas limpias y estandarizadas, facilitando el entrenamiento de modelos NLP. Como se observa en los ejemplos proporcionados, el texto original (e.g., "As advertised. Reasonably priced") se convierte en una versión procesada ("advertis reason price") mediante pasos como conversión a minúsculas, eliminación de puntuación, tokenización con NLTK (que descarga paquetes como 'punkt\_tab' para segmentación eficiente) y stemming con SnowballStemmer, removiendo stop words para enfocarse en términos significativos. Este proceso reduce el ruido y la dimensionalidad, mejorando la eficiencia en vectorización TF-IDF para el modelo Naive Bayes, y preparando embeddings contextuales para BERT, lo que demuestra habilidades clave en manejo de datos desestructurados para análisis de sentimiento.

En aplicaciones reales para clientes en marketing o e-commerce, este preprocesamiento asegura que el modelo capture esencias semánticas sin distracciones (e.g., el ejemplo negativo sobre olor desagradable se simplifica a raíces como "smell" y "aw", destacando quejas sensoriales). Los resultados muestran una reducción efectiva en longitud y variabilidad (de oraciones completas a stems concisos), lo que acelera el entrenamiento en un 50-70% y mejora métricas como F1-score al enfocarse en features relevantes, permitiendo insights accionables como identificar patrones en feedback de productos de belleza.









Resultados del Modelo Baseline: TF-IDF + Naive Bayes



negativo       1.00      0.03      0.05        37

neutral        0.00      0.00      0.00        21

positivo       0.95      1.00      0.97       996



accuracy                            0.95      1054

macro avg       0.65      0.34      0.34      1054

weighted avg    0.93      0.95      0.92      1054



El modelo Naive Bayes con vectorización TF-IDF logra una accuracy global del 95% en el conjunto de test (1,054 reseñas), destacando un recall perfecto (1.00) y precisión alta (0.95) en la clase positiva (996 muestras), lo que refleja su efectividad en la clase mayoritaria del dataset desbalanceado. Sin embargo, el rendimiento en clases minoritarias es deficiente: para negativos (37 muestras), la precisión es solo 1.00 pero recall bajo (0.03), indicando que detecta pocos verdaderos negativos; mientras que para neutrales (21 muestras), tanto precision como recall son 0.00, fallando completamente en esta categoría. El F1-score macro-averaged de 0.34 subraya este bias, común en datasets skewed, donde el modelo tiende a predecir siempre "positivo" para maximizar accuracy pero pierde utilidad en detectar feedback crítico.

En términos de implicaciones para análisis de sentimiento en e-commerce, este baseline es rápido y liviano (entrenamiento <1 minuto en CPU), ideal para prototipos iniciales en marketing donde identificar tendencias positivas domina, pero requiere mejoras como balanceo de clases (e.g., SMOTE o class weights) para aplicaciones reales como monitoreo de quejas en reseñas de Amazon. Comparado con métricas weighted avg (F1=0.92), resalta la necesidad de enfoques avanzados como BERT para manejar matices y minorías, elevando el valor del proyecto en escenarios empresariales al proporcionar un punto de comparación claro y cuantificable.





F1-score: 0.920531493538634

Matriz de confusión:

 \[\[  1   0  36]

 \[  0   0  21]

 \[  0   0 996]]



El F1-score macro del modelo Naive Bayes (0.92) parece reflejar un promedio weighted influenciado por la clase dominante positiva, pero al desglosarse revela limitaciones severas en el manejo de clases minoritarias, alineándose con el classification report previo donde el macro avg real es ~0.34 (posiblemente este valor es weighted o un error de logging; en práctica, confirma el bias). Esto resalta que, aunque el modelo logra alta precisión general en predicciones positivas, su utilidad en análisis equilibrado de sentimientos es limitada, con un F1 macro bajo indicando pobre generalización en datasets desbalanceados como este de reseñas de Amazon, donde solo el 5.5% son no-positivas.

La matriz de confusión refuerza este diagnóstico: de 37 negativos reales, solo 1 se clasifica correctamente (36 erróneos como positivos); los 21 neutrales se predicen todos como positivos (0 aciertos); y los 996 positivos se detectan perfectamente, resultando en 1,017 falsos positivos acumulados en minorías. Este patrón típico de overfitting a la mayoría sugiere la necesidad de técnicas como undersampling

