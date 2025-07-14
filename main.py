from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import json
import os
from openai import OpenAI
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = FastAPI()

# Pydantic model for request
class AuditRequest(BaseModel):
    brand: str
    industry: str
    queries: List[str] = ["What is {brand}?", "Best {industry} brands"]
    desired_description: str = ""  # Optional ideal brand description

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com"
)

# Reused utility functions from Content Intelligence
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

# Sentiment analysis (from LLM Sentiment Analysis add-on)
def analyze_sentiment(response):
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

# Analyze tone (new function for perception)
def analyze_tone(response):
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

# Check factual accuracy (basic, client-provided facts)
def check_accuracy(response, desired_description):
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

# Extract citations from LLM response
def extract_citations(response):
    try:
        soup = BeautifulSoup(response, "html.parser")
        urls = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        return urls if urls else ["No citations found"]
    except:
        return ["No citations found"]

# Main audit endpoint
@app.post("/brand-visibility-audit")
def brand_visibility_audit(request: AuditRequest):
    try:
        # Initialize results
        results = {
            "brand_perception": [],
            "competitor_benchmarking": {},
            "citations": [],
            "topic_clusters": [],
            "framing_suggestions": []
        }

        # Fetch ContentStudio trends for industry context
        api_key = os.getenv("CONTENTSTUDIO_API_KEY", "your_api_key_here")
        trends_url = f"https://api.contentstudio.io/v1/trends?query={request.industry}"
        trends = fetch_contentstudio_trends(trends_url, api_key)

        # Query LLMs (DeepSeek) for brand-related prompts
        llm_responses = []
        for query_template in request.queries:
            query = query_template.format(brand=request.brand, industry=request.industry)
            try:
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": query}],
                    temperature=0.5,
                    max_tokens=500
                )
                response = resp.choices[0].message.content.strip()
                llm_responses.append({
                    "query": query,
                    "response": response,
                    "llm": "DeepSeek"
                })
            except Exception as e:
                llm_responses.append({
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "llm": "DeepSeek"
                })

        # Analyze Brand Perception
        for resp in llm_responses:
            sentiment = analyze_sentiment(resp["response"])
            tone = analyze_tone(resp["response"])
            accuracy = check_accuracy(resp["response"], request.desired_description)
            results["brand_perception"].append({
                "query": resp["query"],
                "response": resp["response"],
                "sentiment": sentiment,
                "tone": tone,
                "accuracy": accuracy
            })

        # Competitor Benchmarking
        competitor_counts = {request.brand: 0}
        for resp in llm_responses:
            if "Best" in resp["query"]:
                competitors = ["Competitor A", "Competitor B"]  # Replace with actual competitors
                for competitor in competitors + [request.brand]:
                    if competitor.lower() in resp["response"].lower():
                        competitor_counts[competitor] = competitor_counts.get(competitor, 0) + 1
        total_mentions = sum(competitor_counts.values())
        results["competitor_benchmarking"] = {
            "share_of_voice": {
                comp: f"{(count/total_mentions*100):.1f}%" if total_mentions else "0%"
                for comp, count in competitor_counts.items()
            }
        }

        # Citation Analysis
        for resp in llm_responses:
            citations = extract_citations(resp["response"])
            results["citations"].extend([{"query": resp["query"], "url": url} for url in citations])

        # Topic Clustering (adapted from Content Cluster Visualization)
        embeddings = [extract_keywords_and_embeddings(resp["response"]) for resp in llm_responses]
        if len(embeddings) >= 2:
            n_clusters = min(max(2, len(embeddings) // 2), 10)
            labels, centroids = cluster_embeddings(np.array(embeddings), n_clusters)
            for c in range(n_clusters):
                cluster_responses = [resp["response"] for i, resp in enumerate(llm_responses) if labels[i] == c]
                if cluster_responses:
                    prompt = f"Summarize these responses about {request.brand}: {', '.join(cluster_responses[:15])} into a 3-5 word topic label in Title Case."
                    try:
                        resp = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.5,
                            max_tokens=10
                        )
                        topic = resp.choices[0].message.content.strip()
                    except:
                        topic = f"Cluster {c}"
                    results["topic_clusters"].append({
                        "cluster": c,
                        "topic": topic,
                        "responses": cluster_responses
                    })

        # Framing Suggestions
        for resp in results["brand_perception"]:
            framing = f"Frame as {resp['tone'].lower()} and correct inaccuracies: {resp['accuracy']}" if "Error" not in resp["accuracy"] else "Correct LLM response errors"
            results["framing_suggestions"].append({
                "query": resp["query"],
                "suggestion": framing
            })

        # Save to CSV for Google Sheets
        df = pd.DataFrame([
            {
                "query": bp["query"],
                "response": bp["response"],
                "sentiment": bp["sentiment"],
                "tone": bp["tone"],
                "accuracy": bp["accuracy"],
                "framing_suggestion": fs["suggestion"]
            }
            for bp, fs in zip(results["brand_perception"], results["framing_suggestions"])
        ])
        df.to_csv("brand_audit_output.csv", index=False)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
