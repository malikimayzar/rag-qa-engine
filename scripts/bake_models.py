from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Baking embedding model...")
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Baking reranker model...")
AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("All models baked!")
