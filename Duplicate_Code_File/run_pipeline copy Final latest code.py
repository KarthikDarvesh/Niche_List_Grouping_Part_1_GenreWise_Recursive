from modules import niche_utils as nu
import pandas as pd
import torch

# ------------------- Fastapi Libraries--------------------------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import json

app = FastAPI()

# Load constants
hf_token = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/niche_grouping/")
async def process_niches(file: UploadFile = File(...)):
    try:
        # ------------------- Load Uploaded JSON File -------------------
        contents = await file.read()
        raw_niches = json.loads(contents)

        # Save temporarily for load_niches() compatibility
        temp_path = "temp_niche_file.json"
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
        
        
        # ------------------- Return JSON Response -------------------
        # It returns the final output of your clustering pipeline as a JSON response in FastAPI.
        return JSONResponse(content=named_clusters_df.to_dict(orient="records"))


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