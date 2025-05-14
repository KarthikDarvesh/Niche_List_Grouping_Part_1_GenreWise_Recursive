from modules import niche_utils as nu
from modules import genre_utils as gu
import pandas as pd
import torch

# ------------------- Fastapi Libraries--------------------------------------
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import json

app = FastAPI()
 
# ------------------- Load Genre List -------------------
genre_path = "genre_list.json"
genres = gu.load_genres(genre_path)  # ✅ Fix: Use gu.load_genres
# Load constants
hf_token = "Keep HF Token Here"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set global seed for reproducibility
nu.set_global_seed(42)

# # Load all models once at startup
# tokenizer, embed_model = nu.load_embedding_model(hf_token)
# name_tokenizer, name_model, name_device = nu.load_naming_model()
# sim_model = nu.load_similarity_model()
# genre_tokenizer, genre_model, _ = gu.load_embedding_model_gener(hf_token)
# cluster_tokenizer, cluster_model, _ = gu.load_embedding_model_gener(hf_token)

@app.post("/niche_grouping/")
async def process_niches(request: Request):
    
    temp_path = "temp_niche_file.json"  # define here so it's accessible in finally
    try:
        # =================================================
        # ------------------- NICHE LIST-------------------
        # =================================================


        # ------------------- Load Uploaded JSON File -------------------
        # contents = await file.read()
        # raw_niches = json.loads(contents)
 
        # # Save temporarily for load_niches() compatibility
        # temp_path = "temp_niche_file.json"
        # with open(temp_path, "w", encoding="utf-8") as f:
        #     json.dump(raw_niches, f)

        # ------------------- Receive JSON Body -------------------
        raw_niches = await request.json()

        # Save temporarily for load_niches() compatibility
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(raw_niches, f)

        # ------------------- Load & Preprocess Niches -------------------
        print("📥 Loading and normalizing niches...")
        unique_niches = nu.load_niches(temp_path)

        # ------------------- Embedding Generation -------------------
        print("🔍 Generating embeddings...")
        tokenizer, embed_model = nu.load_embedding_model(hf_token)
        embeddings = nu.get_embeddings(unique_niches, tokenizer, embed_model, device=device)
        print(f"✅ Generated embeddings. Shape: {embeddings.shape}")
        reduced = nu.reduce_embeddings(embeddings)

        # ------------------- Clustering -------------------
        print("📊 Clustering...")
        cluster_labels, centroids = nu.cluster_niches(reduced)

        # print("🔗 Merging similar clusters...")
        # merged_labels = nu.merge_similar_clusters(centroids, cluster_labels, similarity_threshold=0.90)

        # ------------------- Cluster Naming -------------------
        print("🧠 Generating cluster names using FLAN-T5...")
        name_tokenizer, name_model, _ = nu.load_naming_model()  # device already defined above
        named_clusters = nu.generate_cluster_names(unique_niches, cluster_labels, name_tokenizer, name_model, device)

        # ------------------- Semantic Accuracy -------------------
        print("📐 Calculating semantic similarity scores...")
        sim_model = nu.load_similarity_model()
        named_clusters_df = nu.add_semantic_similarity(pd.DataFrame(named_clusters), sim_model)
        
        # =================================================
        # ------------------- GENER LIST-------------------
        # =================================================

        # ------------------- Embedding genre names and generated cluster names separately -------------------
        # -------- Saprate Genre Embedding --------
        print("🎯 Embedding genres...")
        genre_tokenizer, genre_model, _ = gu.load_embedding_model_gener(hf_token)
        genre_embeddings = gu.get_embeddings_gener(genres, genre_tokenizer, genre_model, device)
        print(f"✅ Generated embeddings. Shape: {genre_embeddings.shape}")
        # reduced_genre_embeddings = gu.reduce_embeddings_gener(genre_embeddings)

        # -------- Saprate Cluster Name Embedding --------
        print("📦 Embedding (generated_cluster_names...)")
        generated_cluster_names = [entry['Generated_Cluster_Name'] for entry in named_clusters]
        cluster_tokenizer, cluster_model, _ = gu.load_embedding_model_gener(hf_token)
        cluster_embeddings = gu.get_embeddings_gener(generated_cluster_names, cluster_tokenizer, cluster_model, device)
        print(f"✅ Generated embeddings. Shape: {cluster_embeddings.shape}")
        # reduced_cluster_embeddings = gu.reduce_embeddings_gener(cluster_embeddings)

        # ------------------- Genre-to-Cluster Mapping -------------------
        print("🔗 Mapping (generated_clusters_name) to genres using cosine similarity")
        genre_cluster_mapping = gu.map_clusters_to_genres_crossencoder(
            generated_cluster_names,
            genres,
            model_name="cross-encoder/stsb-roberta-base"  # or any other supported one
        )
        # ------------------- Unmapped Clusters Debug Check -------------------
        all_mapped_clusters = set(c for clusters in genre_cluster_mapping.values() for c in clusters)
        unmapped_clusters = set(generated_cluster_names) - all_mapped_clusters
        if unmapped_clusters:
            print(f"⚠️ WARNING: {len(unmapped_clusters)} clusters were NOT mapped to any genre!")
        else:
            print("✅ All clusters successfully mapped to genres.")

        # ------------------- Genre–Cluster Similarity Report -------------------
        print("📊 Scoring (genre)-(generated_cluster_name) semantic similarity...")
        genre_similarity_df = gu.compute_genre_cluster_similarity(genre_cluster_mapping, sim_model)

        # ------------------- Final Response -------------------
        return JSONResponse(content={
            "Mapped_Gen_clusters_With_Genres": genre_cluster_mapping,
            "List_of_Gen_Clusters_and_Niches_with_Scores": named_clusters_df.to_dict(orient="records"),
            "List_of_Gen_Clusters_and_Genre_With_Scores": genre_similarity_df.to_dict(orient="records")
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        # ✅ Cleanup happens here even if an error occurs
        # ❓ Why do I need to use os.remove(temp_path) at the end of my FastAPI code?
        # You're doing this because your utility function nu.load_niches(path) expects a file path, not raw JSON. That’s fine.
        # But once you're done with the processing, that temporary file is no longer needed.
        # ✅ Cleans up that temporary file from the server's storage to:
        #    - Prevent clutter and buildup of unused files
        #    - Avoid overwriting issues if another request hits the same endpoint
        #    - Ensure safe, isolated handling of file uploads
        # ❗ What Happens If You Don’t Use It?
        #    - Multiple users hit your API ➝ all requests write to the same temp_niche_file.json
        #    - 💥 Race conditions or overwrite issues
        #    - Server directory gets messy with leftover files
        if os.path.exists(temp_path):
            os.remove(temp_path)
