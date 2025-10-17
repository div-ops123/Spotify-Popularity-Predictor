# IMPORTANT   

# ✅ At preprocessing/training time, you fit the transformers and save them to disk (using pickle).
# ✅ At inference time, you load those transformers and call .transform() only — not .fit_transform().