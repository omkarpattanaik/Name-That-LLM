from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Submitting Version 18 to Name That LLM competition...")
res = api.competition_submit_code(
    file_name='submission.parquet',
    message='V18 Apex Sovereign Calibrated: Calibrated Temperature (--sk_div 0) + 227-Feature Engine + V40A Mass Conservation + Damped Sinkhorn',
    competition='name-that-llm',
    kernel='omkarpattanaik123/name-that-llm-v18-apex-sovereign-calibrated',
    kernel_version=1
)
print("Submission response:", res)

subs = api.competition_submissions('name-that-llm')
print(f"\nTotal submissions found: {len(subs)}")
for s in subs[:6]:
    ref = getattr(s, 'ref', 'N/A')
    date = getattr(s, 'date', 'N/A')
    status = getattr(s, 'status', 'N/A')
    score = getattr(s, 'publicScore', getattr(s, 'public_score', 'N/A'))
    desc = getattr(s, 'description', 'N/A')
    print(f"Ref: {ref} | Date: {date} | Status: {status} | Public Score: {score} | Description: {desc}")
