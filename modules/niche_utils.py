
# ------------------ Imports ------------------
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from umap import UMAP
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------ Global Model References ------------------
tokenizer = None
embed_model = None
name_tokenizer = None
name_model = None
name_device = None
similarity_model = None
# ------------------ Set Global Seed ------------------
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------ Embedding Model ------------------
def load_embedding_model(hf_token, model_name="intfloat/e5-large-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token).to(
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

# ------------------ Cluster Naming ------------------
def load_naming_model():
    model_name = "google/flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return tokenizer, model, device

def pick_best_name(generated_names, cluster_topics):
    if len(generated_names) == 1:
        return generated_names[0]
    cluster_text = " ".join(cluster_topics)
    vectorizer = TfidfVectorizer().fit_transform([cluster_text] + generated_names)
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    best_index = similarity_matrix.argmax()
    return generated_names[best_index]

def get_cluster_name(cluster, tokenizer, model, device):
    if not cluster:
        return "Miscellaneous"
    prompt = f"""
    Analyze the given cluster by understanding its semantic meaning and context.

    ## Task:
    - Generate a **single, concise category name** (3-7 words).
    - The name must **accurately represent the entire cluster** without focusing on only one example.
    - Avoid being **too specific**.
    - Avoid being **too broad**.
    - Use **real-world terminology**.
    - Return ONLY the category name.

    ## Topics in the Cluster:
    {', '.join(cluster[:200])}
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=15,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_names = [name.strip() for name in generated_text.split(",")]
    return pick_best_name(generated_names, cluster)

# ------------------ Similarity Model ------------------
def load_similarity_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

# ------------------ Genre-wise Clustering ------------------
def cluster_niches_by_genre_with_similarity(data, hf_token, n_clusters=20):
    print("📥 Loading and normalizing niches...")

    global tokenizer, embed_model, similarity_model
    global name_tokenizer, name_model, name_device  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embedding model
    if tokenizer is None or embed_model is None:
        print("🔄 Loading embedding model...")
        tokenizer, embed_model = load_embedding_model(hf_token)

    # Load naming model
    if name_tokenizer is None or name_model is None or name_device is None:
        print("🔄 Loading naming model...")
        name_tokenizer, name_model, name_device = load_naming_model()

    # Load similarity model
    if similarity_model is None:
        print("🔄 Loading similarity model...")
        similarity_model = load_similarity_model()


    # Log the size of input dataset
    print(f"📦 Loaded entire dataset with {len(data)} entries.")

    # Extract unique genres and niches using sets
    genre_to_niches = defaultdict(set)
    for entry in data:
        genre = entry["genre_name"].strip()
        niche = entry["niche_name"].strip()
        genre_to_niches[genre].add(niche)

    # Calculate counts BEFORE conversion
    total_genres = len(genre_to_niches)
    total_niches = sum(len(niches) for niches in genre_to_niches.values())
    print(f"📊 Found length of unique genres: {total_genres} and total unique niches: {total_niches}")

    # Convert sets to lists for model processing
    genre_to_niches = {genre: list(niches) for genre, niches in genre_to_niches.items()}

    # Confirm extraction
    print(f"✅ Extracted unique genres and niches. Genres: {total_genres}, Total Unique Niches: {total_niches}")
    print(f"📦 Total unique genre_to_niches: {len(genre_to_niches)}")

    final_results = []

    # Process each genre
    for genre, niches in genre_to_niches.items():
        print(f"\n🔍 Processing genre: {genre} with {len(niches)} unique niches...")
        actual_n_clusters = min(n_clusters, len(niches))

        # Generate embeddings
        print("🔍 Generating embeddings...")
        embeddings = get_embeddings(niches, tokenizer, embed_model, device)
        print(f"✅ Generated embeddings. Shape: {embeddings.shape}")

        
        # Cluster niches using MiniBatchKMeans
        print("📊 Clustering...")
        print(f"📊 Clustering {len(embeddings)} niches into {actual_n_clusters} clusters using Mini-Batch K-Means...")
        minibatchkmeans = MiniBatchKMeans(n_clusters=actual_n_clusters, random_state=42)
        labels = minibatchkmeans.fit_predict(embeddings)
        print(f"✅ Clustering complete. Total clusters: {len(set(labels))}")

        # Map each cluster label to its list of niches
        cluster_map = defaultdict(list)
        for niche, label in zip(niches, labels):
            cluster_map[label].append(niche)

        subclusters = []

        # Naming clusters
        print("🧠 Generating cluster names using FLAN-T5 Large...")
        for cluster_niches in cluster_map.values():
            try:
                cluster_name = get_cluster_name(cluster_niches, name_tokenizer, name_model, name_device)
            except Exception as e:
                print(f"⚠️ Failed to generate cluster name: {str(e)}")
                cluster_name = "Unnamed Cluster"
            

            cluster_emb = similarity_model.encode([cluster_name], convert_to_tensor=True)

            enriched = []
            for niche in cluster_niches:
                niche_emb = similarity_model.encode([niche], convert_to_tensor=True)
                sim = util.cos_sim(cluster_emb, niche_emb).item()
                enriched.append({
                    "Niche_Name": niche,
                    "Semantic_Similarity": round(sim, 4)
                })

            subclusters.append({
                "Generated_Cluster_Name": cluster_name,
                "List_Of_Niches_In_The_Cluster": enriched
            })

        final_results.append({
            "Genre_Name": genre,
            "Subclusters": subclusters
        })

    return final_results

# ------------------ Save Utilities ------------------
def save_to_json(obj, filename="clustered_niches_by_genre.json"):
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
