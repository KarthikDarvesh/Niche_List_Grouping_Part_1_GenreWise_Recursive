# 📚 NICHE GROUPING PROJECT PART 1

---

## Overview

This project is a FastAPI-based microservice designed to cluster niches by genre using modern NLP embeddings, clustering algorithms, and cluster naming models (e.g., FLAN-T5-Large).

It enables semantic grouping of niches based on contextual similarity, and auto-generates meaningful cluster names.

---

## 📁 Data

- **Source**:
  - API Endpoint: Data is stored in a PostgreSQL database, and I created an API to access that data.
  - Postgres Database Table: `genre_niche_mapped_view`

- **Fields Available**:
  - `genre_id` (Integer): Unique ID of the genre.
  - `genre_name` (String): Name of the genre (e.g., Adventure, Lifestyle, etc.).
  - `niche_name` (String): Name of the niche belonging to the genre.

---

## 🔄 Workflow

1. 📥 **Initial Input**:
   - For each iteration, one genre along with all its associated `niches_name` is taken as input.

2. 🔄 **Genre-wise Processing**:
   - Collect all `niches_name` associated with the current genre.
   - Convert the collected `niches_name` into vector embeddings (numerical representations).
   - **Model Used for Embedding**:  
     ➔ `sentence-transformers` → `intfloat/e5-large-v2`

3. 🧩 **Clustering per Genre**:
   - Apply MiniBatchKMeans clustering on the embeddings.
   - **Model Used for Clustering**:  
     ➔ `sklearn.cluster.MiniBatchKMeans`
   - Create 20 clusters for the current genre's niches.

4. 🏷 **Cluster Naming**:
   - Assign a meaningful and descriptive name to each of the 20 clusters based on their central theme.
   - **Model Used for Naming**:  
     ➔ `transformers` → `google/flan-t5-large`

5. 🗂 **Genre-Cluster Mapping**:
   - Map the current genre to:
     - Its 20 generated cluster names.
     - The list of associated `niches_name` under each cluster.
   - **Model Used for Similarity Checking (Optional Enrichment Step)**:  
     ➔ `sentence-transformers` → `all-mpnet-base-v2`

6. 🔁 **Iteration**:
   - Repeat this complete process for each genre individually.
   - At the end, you will have processed all genres (e.g., 88 genres).

7. 📊 **Output Structure**:
   - **Genre** (Example: "Fitness")
     - **Cluster 1 Name** (Example: "Weight Loss Programs")
       - `niches_name 1`
       - `niches_name 2`
       - `niches_name 3`
       - ...
     - **Cluster 2 Name** (Example: "Home Workouts")
       - `niches_name 4`
       - `niches_name 5`
       - `niches_name 6`
       - ...
     - (up to 20 clusters for this genre)

   - **Next Genre** (Example: "Technology")
     - **Cluster 1 Name** (Example: "AI Tools")
       - `niches_name 1`
       - `niches_name 2`
       - `niches_name 3`
       - ...
     - (up to 20 clusters for this genre)
       
   - And so on, until all genres are processed.

---

## Final Summary

- **📥Input**:  
  → 1 genre + its associated niches → vectorize → cluster into 20 → generate cluster names → map clusters + their niches under the genre with (Similarity Score) → repeat for next genre.

- **📤Output**:  
  → Hierarchical structure = Genre → Cluster Name → List of Associated Niches.

---

## 🧰 Libraries Used

- `numpy`
- `pandas`
- `torch`
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `fastapi`
- `json`
- `collections`
- `random`
- `umap-learn`
- `datetime`
- `pytz`
- `uvicorn[standard]`
- `huggingface-hub`
- `accelerate`
- `tokenizers`

---

## 🗂️ Project Directory Structure

![image](https://github.com/user-attachments/assets/020d21fc-e7f7-40e9-a10f-947052029998)

---

## 🌐 API Endpoint Details

- **Route**: `/predict`
- **Method**: `GET`
- **Test URL**: `http://localhost:8001/niche_grouping/`
- **Content-Type**: `application/json`
  
- **Body**: `Json-Array Input:`
### 📥 Input Body Example
```
[
  {
    "genre_id": 1,
    "genre_name": "Adventure",
    "niche_id": 872,
    "niche_name": "Offbeat Destinations"
  },
  {
    "genre_id": 1,
    "genre_name": "Adventure",
    "niche_id": 873,
    "niche_name": "High-Altitude Exploration"
  },
  .
  .
  {
    "genre_id": 1,
    "genre_name": "Adventure",
    "niche_id": 882,
    "niche_name": "Travel"
  }
  .
  .
  .
]
```
### 📤 API Response Format
```
- **Genre** (Example: "Fitness")
  - **Cluster 1 Name** (Example: "Weight Loss Programs")
    - niches_name 1
    - niches_name 2
    - niches_name 3
    - ...
  - **Cluster 2 Name** (Example: "Home Workouts")
    - niches_name 4
    - niches_name 5
    - niches_name 6
    - ...
  - ... (up to 20 clusters per genre)

- **Next Genre** (Example: "Technology")
  - **Cluster 1 Name** (Example: "AI Tools")
    - niches_name 1
    - niches_name 2
    - niches_name 3
    - ...
  - ... (up to 20 clusters per genre)
```
---

## 🗄️ Database

- After API processing (for each genre), the processed output is saved to a PostgreSQL table named **`channel_pillar`**.

---

## ▶️ Run the API

  ### 📦 Install Required Libraries
  ```
  pip install -r requirements.txt
  ```
  
  ### 🐳 Build Docker Image
  ```
  docker build -t niche_grouping .
  ```
  
  ### 🐳 Run Docker Container
  ```
  docker run --gpus all -p 8001:8001 niche_grouping
  ```

  ### 🚀 API Hit
  ```
  http://localhost:8001/niche_grouping/
  ```

