"""
Quick script to inspect the structure of preprocessed_data.pkl
"""
import pickle
import sys

data_path = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\data\preprocessed_data.pkl"

print("Loading pickle file...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"\nData type: {type(data)}")
print(f"\nKeys in data: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

if isinstance(data, dict):
    for key in data.keys():
        value = data[key]
        if hasattr(value, 'shape'):
            print(f"\n{key}: shape={value.shape}, dtype={value.dtype}")
        elif hasattr(value, '__len__'):
            print(f"\n{key}: len={len(value)}, type={type(value)}")
        else:
            print(f"\n{key}: {type(value)}")
