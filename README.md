# PP10
Practice Project 10, Sentiment Analysis

# BERT Transformers for Sentiment Analysis on News Data
This project utilizes BERT transformers for sentiment analysis on news articles. The model is trained on the IMDB dataset, which is then used to predict sentiments of news articles related to a specific topic. The following guide demonstrates how to set up the project and use the pre-trained model for sentiment analysis.

BERT Model Architecture must be downloaded from transformers and must be compiled before loading weights.
```Python
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

load_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

load_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
load_model.load_weights('model_weights')
```
Index:

https://drive.google.com/file/d/1fFwX6ISCZhXjyFxjei-RVYnlJ6LxBw5Y/view

Weights:

https://drive.google.com/file/d/1oelRnAixYd0ol3C1zupmmIAOEqANPud0/view

# Conclusion
This project demonstrates the power of transfer learning and the BERT classification model for sentiment analysis. The model performs well on both the IMDB dataset and real-world news articles. By saving and loading model weights, the training process becomes more efficient and allows for further experimentation. However, training time can still be lengthy, and future work could explore faster and simpler models for similar tasks.
