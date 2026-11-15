"""Lab 1 extra: same model, different chunks (IT help-desk tickets).

Run: python compare.py
Before running, predict which pairs score highest and lowest.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

TICKETS = [
    "VPN keeps disconnecting when I work from home",
    "Cannot reach the company network remotely",
    "I was locked out of my account after too many attempts",
    "Please reset my password",
    "The printer on floor 3 is out of toner",
    "Requesting a new laptop for a new hire",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(TICKETS, normalize_embeddings=True)
scores = vectors @ vectors.T   # every ticket against every other ticket

print("Similarity matrix (rows and columns are tickets 1 to 6):\n")
print("      " + "".join(f"  T{i + 1}  " for i in range(len(TICKETS))))
for i, row in enumerate(scores):
    print(f"T{i + 1}  " + "".join(f"{s:6.2f} " for s in row))

print("\nTickets:")
for i, t in enumerate(TICKETS):
    print(f"  T{i + 1}: {t}")

print("\nClosest pair for each ticket:")
for i in range(len(TICKETS)):
    others = [(scores[i][j], j) for j in range(len(TICKETS)) if j != i]
    s, j = max(others)
    print(f"  T{i + 1} -> T{j + 1}  ({s:.2f})")
