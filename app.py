from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
bacteria_df = pd.read_csv("final_dataset/bacteria_k5_wide.csv")
phage_df = pd.read_csv("final_dataset/phage_k5_wide.csv")

bacteria_names = bacteria_df["genome_name"]
phage_names = phage_df["genome_name"]

X_bac = bacteria_df.select_dtypes(include=["number"])
X_phage = phage_df.select_dtypes(include=["number"])

common_kmers = X_bac.columns.intersection(X_phage.columns)
X_bac = X_bac[common_kmers]
X_phage = X_phage[common_kmers]

similarity = cosine_similarity(X_bac, X_phage)
sim_df = pd.DataFrame(similarity, index=bacteria_names, columns=phage_names)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        bacterium = request.form["bacterium"]
        scores = sim_df.loc[bacterium].sort_values(ascending=False)

        best_phage = scores.index[0]
        best_score = scores.iloc[0]

        # -------- Graph 1: Bar plot (all phages)
        plt.figure(figsize=(9, 4))
        scores.plot(kind="bar")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Phage Similarity for {bacterium}")
        plt.tight_layout()
        plt.savefig("static/bar_all.png")
        plt.close()

        # -------- Graph 2: Top 3 phages
        plt.figure(figsize=(6, 4))
        scores.head(3).plot(kind="bar", color="green")
        plt.ylabel("Cosine Similarity")
        plt.title("Top 3 Phage Candidates")
        plt.tight_layout()
        plt.savefig("static/bar_top3.png")
        plt.close()

        # -------- Graph 3: Global heatmap
        plt.figure(figsize=(8, 5))
        plt.imshow(sim_df.values, aspect="auto")
        plt.colorbar(label="Cosine Similarity")
        plt.xticks(range(len(phage_names)), phage_names, rotation=90)
        plt.yticks(range(len(bacteria_names)), bacteria_names)
        plt.title("Bacteria–Phage Similarity Heatmap (k=5)")
        plt.tight_layout()
        plt.savefig("static/heatmap.png")
        plt.close()

        # -------- Explanation text
        explanation = f"""
        The selected phage ({best_phage}) shows the highest cosine similarity
        score ({best_score:.3f}) among all candidate phages for {bacterium}.
        This indicates closer genomic composition based on k-mer frequency
        profiles. Compared to other phages, {best_phage} consistently ranks
        highest, suggesting stronger potential compatibility.
        """

        result = {
            "bacterium": bacterium,
            "best_phage": best_phage,
            "score": round(best_score, 3),
            "explanation": explanation
        }

    return render_template(
        "index.html",
        bacteria=bacteria_names,
        result=result
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

