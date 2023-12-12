import json
import pandas as pd

# Load JSON data from file
with open('abc.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Flatten the data
flattened_data = []
for entry in data:
    input_text = entry['input']
    for output in entry['output']:
        flattened_data.append({'input': input_text, 'output': output})

# Create a DataFrame
df = pd.DataFrame(flattened_data)

print(df)
