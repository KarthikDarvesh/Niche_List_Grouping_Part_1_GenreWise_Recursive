# ------------------ Imports ------------------
import os
import json
import re
import torch
import numpy as np
import pandas as pd
from collections import Counter

from collections import defaultdict
from umap import UMAP  # ✅ Replaced cuml with CPU version
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM)
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Data Loading & Preprocessing ---------------------
def load_niches(filepath):
    """
    Load and normalize niche names from JSON file.
    Extracts 'niche_name' values, normalizes them, and builds frequency stats.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_niches = json.load(f)

    # Step 1: Extract only the 'niche_name' values
    all_niches = [n["niche_name"] for n in raw_niches if isinstance(n, dict) and "niche_name" in n]
    print(f"📥 Loaded {len(all_niches)} niche entries.")

    # Step 2: Normalize the niches
    def normalize_niche(niche):
        """Normalize a niche string by trimming, lowercasing, and removing extra spaces"""
        return re.sub(r'\s+', ' ', niche.strip().lower())
    # Normalize and count frequencies
    normalized_niches = [normalize_niche(niche) for niche in all_niches]
    frequency_map = Counter(normalized_niches)
    # Create unique list of niches
    unique_niches = list(frequency_map.keys())

    print(f"🔎 Found {len(unique_niches)} unique niches after normalization.")
    # print(f"🧾 Sample normalized niches: {unique_niches[:100]}")

    return unique_niches  # These will be used for embedding

# ------------------ Embedding -----------------------
def load_embedding_model(hf_token, model_name="intfloat/e5-large-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return tokenizer, model

def get_embeddings(texts, tokenizer, model, device="cuda", batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        attention_mask = encoded['attention_mask']
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

def reduce_embeddings(embeddings):
    """
    Reduce embedding dimensionality using UMAP (CPU version).
    """
    print("🧠 Reducing embedding dimensions with UMAP (CPU)...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    umap_model = UMAP(n_components=100, random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    print(f"✅ UMAP Reduction Complete. Shape: {reduced_embeddings.shape}")
    return reduced_embeddings

# ------------------ Clustering ----------------------
def cluster_niches(embeddings: np.ndarray, n_clusters=1500, batch_size=1024):
    """
    Perform Mini-Batch KMeans clustering on the reduced embeddings.
    Returns cluster labels and cluster centroids.
    """
    print(f"📊 Clustering {len(embeddings)} niches into {n_clusters} clusters using Mini-Batch K-Means...")

    # ✅ Step 1: Normalize embeddings (important for cosine similarity)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # ✅ Step 2: Initialize and fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        max_iter=300,
        n_init='auto',
        verbose=0
    )
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    cluster_centroids = kmeans.cluster_centers_

    print(f"✅ Clustering complete. Total clusters: {len(set(cluster_labels))}")
    return cluster_labels, cluster_centroids

# ------------------ merge_similar_clusters ----------------------
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# def merge_similar_clusters(centroids, labels, similarity_threshold=0.98):
#     """
#     Safely merge only highly similar clusters based on cosine similarity.
#     Prevents chain-merging that collapses everything into one cluster.
#     """
#     print("🔗 Merging similar clusters (safely)...")
#     similarity_matrix = cosine_similarity(centroids)
#     n_clusters = len(centroids)
#     cluster_mapping = {i: i for i in range(n_clusters)}
#     merged = set()

#     for i in range(n_clusters):
#         for j in range(i + 1, n_clusters):
#             if similarity_matrix[i][j] > similarity_threshold:
#                 # Only merge if neither has already been merged to something else
#                 if cluster_mapping[i] == i and cluster_mapping[j] == j:
#                     cluster_mapping[j] = i
#                     merged.add((j, i))

#     # Apply mapping to labels
#     new_labels = np.array([cluster_mapping.get(l, l) for l in labels])

#     # Optional: Compact the label space (e.g., convert [0,0,2,3,3] → [0,0,1,2,2])
#     unique_ids = {id_: idx for idx, id_ in enumerate(sorted(set(new_labels)))}
#     new_labels = np.array([unique_ids[l] for l in new_labels])

#     # Logging
#     if merged:
#         print(f"🔍 Merged {len(merged)} pairs of similar clusters.")
#         print(f"✅ Merging complete. Final number of clusters: {len(set(new_labels))}")
#     else:
#         print("ℹ️ No clusters were merged based on the threshold.")

#     return new_labels

# ------------------ Cluster Naming (FLAN-T5) -------------------
def load_naming_model():
    model_name = "google/flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return tokenizer, model, device

# ✅ Generate a name for a single cluster
def get_cluster_name(cluster, tokenizer, model, device):
    
    if not cluster:
        return "Miscellaneous"

    # prompt = f"Generate a short and meaningful category name for the following topics (at least 2 words): {', '.join(cluster[:200])}."
    prompt = f"Given the following list of related topics, generate a short and meaningful (strictly only one) cluster name that best represents the overall theme, cluster name should be concise, relevant: {', '.join(cluster[:200])}."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20, num_beams=5)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Generate names for all clusters
def generate_cluster_names(niches, labels, tokenizer, model, device, batch_size=5):
    clusters = defaultdict(list)
    for niche, label in zip(niches, labels):
        clusters[label].append(niche)

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    named_clusters = []
    for i in range(0, len(sorted_clusters), batch_size):
        batch = sorted_clusters[i:i+batch_size]
        for label, cluster in batch:
            name = get_cluster_name(cluster, tokenizer, model, device)
            named_clusters.append({
                "Generated_Cluster_Name": name,
                "List_Of_Niches_In_The_Cluster": cluster,
                "Number_Of_Niches_In_The_Cluster": len(cluster)
            })

    return named_clusters

# ------------------ Semantic Similarity -------------------
def load_similarity_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def compute_semantic_similarity(cluster_name, niche_list, model):
    if not isinstance(niche_list, list):
        try:
            niche_list = eval(niche_list)
        except:
            niche_list = [niche_list]
    text = ' '.join(niche_list)
    embeddings = model.encode([cluster_name, text], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def add_semantic_similarity(df, model):
    df['Semantic_Similarity'] = df.apply(
        lambda row: compute_semantic_similarity(
            row['Generated_Cluster_Name'],
            row['List_Of_Niches_In_The_Cluster'],
            model
        ),
        axis=1
    )
    return df

# ------------------ Save Utilities -------------------
def save_to_csv(clusters, filename="merged_clustered_niches.csv"):
    df = pd.DataFrame([
        {"Cluster": label, "Niches": ", ".join(niches), "Cluster Size": len(niches)}
        for label, niches in clusters
    ])
    df.to_csv(filename, index=False)
    print(f"✅ Saved to {filename}")

def save_to_json(obj, filename="merged_niches_result.json"):
    def convert_numpy(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, list): return [convert_numpy(i) for i in o]
        if isinstance(o, dict): return {k: convert_numpy(v) for k, v in o.items()}
        return o
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(obj), f, indent=2)
    print(f"✅ JSON saved to {filename}")
