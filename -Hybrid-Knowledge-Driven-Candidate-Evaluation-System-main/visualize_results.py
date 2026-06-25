import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("outputs/final_ranked_candidates.csv")

# -------- Plot 1: Final Score Distribution --------
plt.figure()
plt.hist(df["Final_Score"], bins=20)
plt.title("Final Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Candidates")
plt.show()

# -------- Plot 2: Top 10 Candidates --------
top10 = df.head(10)
plt.figure()
plt.bar(range(1, 11), top10["Final_Score"])
plt.title("Top 10 Ranked Candidates")
plt.xlabel("Rank")
plt.ylabel("Final Score")
plt.show()

# -------- Plot 3: ML vs Rule Contribution --------
plt.figure()
plt.scatter(df["ML_Score"], df["Rule_Score"])
plt.title("ML Score vs Rule Score")
plt.xlabel("ML Score")
plt.ylabel("Rule Score")
plt.show()
