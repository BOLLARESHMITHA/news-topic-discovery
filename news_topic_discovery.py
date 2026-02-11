'''

🟣 “News Topic Discovery Dashboard”
Using Hierarchical Clustering
1️⃣ App Title & Description
Title:
🟣 “News Topic Discovery Dashboard”
Short Description:
“This system uses Hierarchical Clustering to automatically group similar news articles based on textual similarity.”
Purpose (implicit for students):
👉 Discover hidden themes without defining categories upfront.
2️⃣ Input Section (Sidebar – Mandatory)
The sidebar should allow users to control how hierarchical clustering is performed.
📂 Dataset Handling
Upload CSV file (or pre-load Kaggle dataset)
Automatically detect:
Text column (e.g., news, text, headline)
📝 Text Vectorization Controls
User should select:
• Maximum TF-IDF Features
Slider (100 to 2000)
Default: 1000
• Stop Words Removal
Checkbox (Use English stopwords)
• N-gram Range (Optional Bonus)
Dropdown:
Unigrams
Bigrams
Unigrams + Bigrams
🌳 Hierarchical Clustering Controls
• Linkage Method
Dropdown:
Ward
Complete
Average
Single
• Distance Metric
Dropdown:
Euclidean
• Number of Articles for Dendrogram
Slider (20 – 200)
(Important: Dendrogram for subset only)
3️⃣ Clustering Control (Important)
Add a button:
🟦 “Generate Dendrogram”
This ensures students understand:
Hierarchical clustering builds a tree first
We decide cluster count after inspecting tree
After dendrogram inspection:
🟩 “Apply Clustering”
User selects:
• Number of Clusters (based on dendrogram)
4️⃣ Dendrogram Section (Main Panel – Core Part)
Display:
Dendrogram visualization
Y-axis: Distance
X-axis: Article index
Students must visually inspect:
Large vertical gaps
Natural separation levels
Optional:
Allow drawing horizontal cut line (slider for height)
5️⃣ Clustering Visualization Section
Since text is high dimensional:
Project clusters using:
• PCA (2D reduction)
Display:
2D scatter plot
Points colored by cluster
Hover shows sample article snippet
This updates dynamically when:
Linkage changes
Number of clusters changes
6️⃣ Cluster Summary Section (Business View)
Display a table:
Cluster ID	Number of Articles	Top Keywords
For each cluster:
Extract top 10 TF-IDF terms
Show most representative article snippet
This helps editors understand:
What topic each cluster represents
7️⃣ Validation Section 
Since no labels exist, display:
📊 Silhouette Score
Show:
Numeric value
Short explanation:
Close to 1 → well-separated clusters
Close to 0 → overlapping clusters
Negative → poor clustering
Add optional:
Silhouette score comparison for different linkage methods
8️⃣ Business Interpretation Section (Human Language Only)
Below results, display insights like:
🟣 Cluster 0: Articles related to financial markets and stock performance
🟡 Cluster 1: Corporate earnings and quarterly results
🔵 Cluster 2: Economic policy and government announcements
Language must:
Avoid ML terminology
Focus on editorial usefulness
9️⃣ User Guidance / Insight Box
Display:
“Articles grouped in the same cluster share similar vocabulary and themes. These clusters can be used for automatic tagging, recommendations, and content organization.”

'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.markdown("""
This system uses **Hierarchical Clustering** to automatically group similar news articles based on textual similarity.
""")

# --------------------------------------------------------
# SIDEBAR - INPUT CONTROLS
# --------------------------------------------------------
st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    st.sidebar.success("Dataset Loaded Successfully!")
else:
    st.sidebar.warning("Upload a CSV file to proceed.")
    st.stop()

# Detect text columns automatically
text_columns = [col for col in df.columns if df[col].dtype == "object"]

if not text_columns:
    st.error("No text column detected!")
    st.stop()

text_column = st.sidebar.selectbox("Select Text Column", text_columns)

# --------------------------------------------------------
# TEXT VECTORIZATION CONTROLS
# --------------------------------------------------------
st.sidebar.header("📝 Text Vectorization Controls")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    min_value=100,
    max_value=2000,
    value=1000
)

use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# --------------------------------------------------------
# HIERARCHICAL CLUSTERING CONTROLS
# --------------------------------------------------------
st.sidebar.header("🌳 Hierarchical Clustering Controls")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"]
)

subset_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    min_value=20,
    max_value=200,
    value=50
)

# --------------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------------
text_data = df[text_column].astype(str)

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(text_data)

# Dimensionality Reduction (Important for Ward)
n_features = X.shape[1]

if n_features > 1:
    n_components = min(100, n_features - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
else:
    X_reduced = X.toarray()


# --------------------------------------------------------
# GENERATE DENDROGRAM
# --------------------------------------------------------
st.header("🌳 Dendrogram Visualization")

if st.button("🟦 Generate Dendrogram"):

    X_subset = X_reduced[:subset_size]

    Z = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax)
    ax.set_title("Dendrogram for News Articles")
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Distance")

    st.pyplot(fig)

    st.info("Inspect large vertical gaps to determine natural cluster separation.")

# --------------------------------------------------------
# APPLY CLUSTERING
# --------------------------------------------------------
st.header("🟩 Apply Clustering")

num_clusters = st.number_input(
    "Select Number of Clusters",
    min_value=2,
    max_value=15,
    value=3
)

if st.button("Apply Clustering"):

    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X_reduced)

    df["Cluster"] = clusters

    # --------------------------------------------------------
    # PCA VISUALIZATION
    # --------------------------------------------------------
    st.subheader("📊 2D Cluster Visualization (PCA Projection)")

    pca = PCA(n_components=2, random_state=42)
    X_2D = pca.fit_transform(X_reduced)

    plot_df = pd.DataFrame({
        "PCA1": X_2D[:, 0],
        "PCA2": X_2D[:, 1],
        "Cluster": clusters,
        "Snippet": text_data.str[:150]
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color=plot_df["Cluster"].astype(str),
        hover_data=["Snippet"],
        title="Cluster Projection"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # CLUSTER SUMMARY TABLE
    # --------------------------------------------------------
    st.subheader("📋 Cluster Summary")

    terms = vectorizer.get_feature_names_out()
    summary_data = []

    for cluster_id in range(num_clusters):
        cluster_docs = X[clusters == cluster_id]

        if cluster_docs.shape[0] > 0:
            mean_tfidf = cluster_docs.mean(axis=0)
            top_indices = np.asarray(mean_tfidf).flatten().argsort()[-10:]
            top_keywords = [terms[i] for i in top_indices]

            representative_article = df[df["Cluster"] == cluster_id][text_column].iloc[0][:200]

            summary_data.append({
                "Cluster ID": cluster_id,
                "Number of Articles": cluster_docs.shape[0],
                "Top Keywords": ", ".join(top_keywords),
                "Representative Snippet": representative_article
            })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    # --------------------------------------------------------
    # SILHOUETTE SCORE
    # --------------------------------------------------------
    st.subheader("📊 Validation")

    sil_score = silhouette_score(X_reduced, clusters)

    st.metric("Silhouette Score", round(sil_score, 3))

    if sil_score > 0.5:
        st.success("Clusters are well separated.")
    elif sil_score > 0:
        st.warning("Clusters have moderate overlap.")
    else:
        st.error("Poor clustering detected.")

    # --------------------------------------------------------
    # BUSINESS INTERPRETATION
    # --------------------------------------------------------
    st.subheader("🧠 Business Interpretation")

    for row in summary_data:
        st.markdown(f"""
        🔵 **Cluster {row['Cluster ID']}**  
        Articles mainly discuss topics related to:  
        *{row['Top Keywords']}*
        """)

    # --------------------------------------------------------
    # INSIGHT BOX
    # --------------------------------------------------------
    st.info("""
    Articles grouped in the same cluster share similar vocabulary and themes. 
    These clusters can be used for automatic tagging, recommendations, and content organization.
    """)