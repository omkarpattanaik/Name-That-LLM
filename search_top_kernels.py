import kaggle

print("=== Top kernels by vote count ===")
ks_votes = kaggle.api.kernels_list(competition='name-that-llm', sort_by='voteCount', page_size=25)
for k in ks_votes:
    print(f"{k.ref:65s} | votes: {k.total_votes}")

print("\n=== Top kernels by hotness / score ===")
ks_hot = kaggle.api.kernels_list(competition='name-that-llm', sort_by='hotness', page_size=25)
for k in ks_hot:
    print(f"{k.ref:65s} | votes: {k.total_votes}")

