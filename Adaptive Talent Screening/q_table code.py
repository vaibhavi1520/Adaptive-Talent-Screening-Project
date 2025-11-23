import pickle
import os

q_table_file = 'q_table_20250910_135602.pkl'

if os.path.exists(q_table_file):
    with open(q_table_file, 'rb') as f:
        q_table = pickle.load(f)
    print("Successfully loaded the Q-table.")
    print(q_table)
else:
    print(f"Error: The file {q_table_file} was not found.")
