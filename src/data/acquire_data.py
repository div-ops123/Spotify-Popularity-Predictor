"""
So our first goal is to write a clean, production-ready Python script that:

Reads both CSV files safely

Validates them (check for empty or malformed data)

Merges them into a single dataset

Saves the merged file into data/processed/spotify_combined.csv

📄 File: src/data/acquire_data.py

We’ll follow your checklist strictly — testing edge cases, adding clear docstrings, and keeping functions short and focused.

Here’s the step-by-step plan before you code:

🧠 Step-by-Step Plan

Define custom exceptions for common data errors (like missing files or empty datasets).

Create a function load_csv(filepath) — loads one CSV and validates it.

Create a function combine_datasets(df1, df2) — merges both DataFrames.

Create a function save_dataset(df, filepath) — saves the final combined file safely.

Create a main acquire_data() function — orchestrates the above steps.

Add logging messages so you can track progress during automation."""