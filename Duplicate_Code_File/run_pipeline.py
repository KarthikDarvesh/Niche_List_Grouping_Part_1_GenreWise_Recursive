from modules import niche_utils as nu
import pandas as pd
import torch

def main():
    # ------------------- Constants -------------------
    json_path = "/home/karthik22/ML_Project/Niche_List_Grouping/niche_list.json"
    hf_token = "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"  # Replace with your Hugging Face token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------- Load & Preprocess Niches -------------------
    print("📥 Loading and normalizing niches...")
    unique_niches = nu.load_niches(json_path)

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
    name_tokenizer, name_model, device = nu.load_naming_model()  # device already defined above
    named_clusters = nu.generate_cluster_names(unique_niches, cluster_labels, name_tokenizer, name_model, device)

    # ------------------- Semantic Accuracy -------------------
    print("📐 Calculating semantic similarity scores...")
    sim_model = nu.load_similarity_model()
    named_clusters_df = nu.add_semantic_similarity(pd.DataFrame(named_clusters), sim_model)

    # ------------------- Save Outputs -------------------
    print("💾 Saving results...")
    named_clusters_df.to_csv("cluster_name_semantic_accuracy.csv", index=False)
    nu.save_to_json(named_clusters, "merged_niches_result.json")

    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
