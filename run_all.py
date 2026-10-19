import subprocess

PYTHON = r"D:\anaconda3\envs\spark_env\python.exe"

NOTEBOOKS = [
    "01-aharkane-chargement-données.ipynb",
    "02-aharkane-pivot.ipynb",
    "03-aharkane-feature-engineering.ipynb",
    "04-aharkane-label-conformite.ipynb",
    "05-aharkane-Regression_Prediction-Rendement.ipynb",
    "06-aharkane-Clustering_Stations.ipynb",
    "07-aharkane-Synthese_Croisee.ipynb",
]

for nb in NOTEBOOKS:
    print(f"▶️  Lancement de {nb}...")
    subprocess.run([
        PYTHON, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"notebooks/{nb}",
        "--output", nb,
    ], check=True)
    print(f"✅ {nb} terminé\n")

print("🎉 Pipeline complet exécuté.")