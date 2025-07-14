import pandas as pd
import numpy as np
import umap
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com"
)

def parse_embedding(s):
    """Parse embedding string from CSV into numpy array."""
    try:
        parts = [float(x) for x in s.split(",") if x]
        return np.array(parts)
    except:
        return None

def analyze_sentiment(response):
    """Reuse LLM Sentiment Analysis add-on logic."""
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

def generate_cluster_label(responses):
    """Generate 3-5 word topic label for a cluster using DeepSeek."""
    prompt = f"Summarize these responses into a 3-5 word topic label in Title Case: {', '.join(responses[:15])}"
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except:
        return "Unnamed Cluster"

def visualize_brand_topics(input_csv="brand_audit_output.csv", output_csv="brand_visualization_output.csv", output_html="brand_visualization.html"):
    """Generate 3D visualization of brand-topic relationships from LLM responses."""
    # Load CSV from main.py
    df = pd.read_csv(input_csv)
    print("Raw columns:", df.columns.tolist())

    # Clean and validate columns
    required_cols = ["query", "response", "llm", "embedding", "sentiment", "framing_suggestion"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns.tolist()}")

    # Parse embeddings
    expected_dim = 384  # all-MiniLM-L6-v2 dimension (from main.py)
    df["embedding"] = df["embedding"].apply(parse_embedding)
    df = df.dropna(subset=["embedding"])
    if len(df) < 2:
        raise ValueError("Insufficient valid embeddings for clustering")
    embeddings = np.vstack(df["embedding"].values)
    print(f"Parsed embeddings: {len(df)}")

    # K-means clustering
    sil_scores = {}
    for k in range(2, min(20, len(df))):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)
        sil_scores[k] = silhouette_score(embeddings, labels)
    best_k = max(sil_scores, key=sil_scores.get, default=2)
    if best_k < 2:
        best_k = 2
    df["cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings)
    print(f"Chosen clusters: {best_k}")

    # UMAP 3D reduction
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, len(df)-1),
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    df[["x", "y", "z"]] = coords

    # Alignment scoring (based on desired description or centroid)
    desired_description = df["framing_suggestion"].iloc[0] if df["framing_suggestion"].notna().any() else "Generic brand description"
    desired_embedding = extract_keywords_and_embeddings(desired_description)  # From main.py
    df["alignment_score"] = cosine_similarity(embeddings, [desired_embedding]).flatten()
    cutoff = np.percentile(df["alignment_score"], 10)  # Bottom 10% are misaligned
    df["misaligned"] = df["alignment_score"] < cutoff
    print(f"Misaligned count: {df['misaligned'].sum()}")

    # Generate cluster labels
    cluster_topics = {}
    for c in sorted(df["cluster"].unique()):
        cluster_responses = df.loc[df["cluster"] == c, "response"].tolist()
        if cluster_responses:
            cluster_topics[c] = generate_cluster_label(cluster_responses)
            print(f"Cluster {c}: {cluster_topics[c]}")
    df["cluster_topic"] = df["cluster"].map(cluster_topics)

    # Categorize by sentiment or misalignment
    df["plot_cat"] = df["sentiment"]
    df.loc[df["misaligned"], "plot_cat"] = "misaligned"

    # Generate 3D plot
    fig = go.Figure()
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        col = colors[c % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=sub.x, y=sub.y, z=sub.z,
            mode="markers",
            marker=dict(size=4, color=col),
            name=f"{cluster_topics[c]} ({len(sub)})",
            text=sub["query"] + ": " + sub["response"],
            hovertemplate="%{text}<extra></extra>"
        ))
    for cat, sym, nm in [
        ("positive", "circle", "Positive"),
        ("negative", "x", "Negative"),
        ("neutral", "square", "Neutral"),
        ("misaligned", "diamond", "Misaligned")
    ]:
        part = df[df["plot_cat"] == cat]
        if not part.empty:
            fig.add_trace(go.Scatter3d(
                x=part.x, y=part.y, z=part.z,
                mode="markers",
                marker=dict(size=6, symbol=sym),
                name=f"{nm} ({len(part)})",
                text=part["query"] + ": " + part["response"],
                hovertemplate="%{text}<extra></extra>"
            ))
    fig.update_layout(
        title=f"Brand Topic Clustering for {df['query'].iloc[0].split()[2]}",
        scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
        width=1200, height=600
    )
    fig.write_html(output_html)
    print(f"Saved visualization: {output_html}")

    # Save results for Google Sheets
    output_cols = ["query", "response", "llm", "cluster", "cluster_topic", "sentiment", "alignment_score", "misaligned", "framing_suggestion"]
    df[output_cols].to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """Reuse from main.py for embedding desired description."""
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])[0]
    return embedding

if __name__ == "__main__":
    visualize_brand_topics()
