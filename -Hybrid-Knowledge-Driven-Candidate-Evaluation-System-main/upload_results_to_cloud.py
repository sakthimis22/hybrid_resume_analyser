import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# ---------------- FIREBASE INIT ----------------
cred = credentials.Certificate("config/firebase_key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------- LOAD FINAL RESULTS ----------------
results_df = pd.read_csv("outputs/final_ranked_candidates.csv")

# ---------------- UPLOAD TO FIRESTORE ----------------
collection_ref = db.collection("ranked_candidates")

for idx, row in results_df.iterrows():
    collection_ref.add({
        "resume_id": int(row["Resume_ID"]),
        "ml_score": float(row["ML_Score"]),
        "rule_score": float(row["Rule_Score"]),
        "final_score": float(row["Final_Score"]),
        "rank": int(idx + 1)
    })

print("✅ Results successfully uploaded to Firebase Firestore")
