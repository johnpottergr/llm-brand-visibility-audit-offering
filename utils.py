import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI

def fetch_contentstudio_trends(url, api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        return [item.get("title", "trend") for item in response.json().get("data", [])]
    except Exception as e:
        print(f"Error fetching trends: {e}")
        return []

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])[0]
    return embedding

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def analyze_sentiment(response, client):
    prompt = f"Analyze sentiment of: {response}. Return positive, negative, or neutral."
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_tone(response, client):
    prompt = f"Analyze tone of: {response}. Return authoritative, casual, technical, or misleading."
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def check_accuracy(response, desired_description, client):
    if not desired_description:
        return "No desired description provided"
    prompt = f"Compare: {response} with facts: {desired_description}. List any inaccuracies."
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extract_citations(response):
    try:
        soup = BeautifulSoup(response, "html.parser")
        urls = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        return urls if urls else ["No citations found"]
    except:
        return ["No citations found"]
