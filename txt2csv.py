import re
import pandas as pd
import requests 
from datasets import Dataset, load_dataset
from huggingface_hub import Repository

# Set the Google Docs file ID
file_id = '1ih00ucjcgCSOkhfx82ZhvxA_G-oNyhHAqZmx2rytHFU'

# Set the endpoint URL for the Google Drive API
url = f'https://docs.google.com/document/d/{file_id}/export?format=txt'

pattern = r'([a-zA-Z\s]+):(.+)'

print('Downloading The Document')
try:
    response = requests.get(url)
    print('Document Download Complete!')
except:
    print("Document Download Failed!")
    print("Solutions: Check Network Connection, Check url and file id")

data = {
    'name': [],
    'line': []
}

with open('Mahiru Shiina dataset.txt', 'rt') as file:
  for line in file.readlines():
    match = re.findall(pattern, line)
    if match:
      name, line = match[0]
      data['name'].append(name)
      data['line'].append(line)
print('Processing Complete!')

df = pd.DataFrame(data)

df.head()

print(f"Every Mahiru Dialog :{sum(df['name'] == 'Mahiru')}")
print(f"Every Dialog :{len(df)}")
print('Writing CSV')

df.to_csv('mahiru_shiina.csv', index=False)

print('Pushing Dataset to Repo')
with open('HuggingFace-API-key.txt', 'rt') as f:
  HUGGINGFACE_API_KEY = f.read().strip()
dataset = load_dataset("csv", data_files="mahiru_shiina.csv")
dataset.push_to_hub(repo_id='Mahiru-Proto', token=HUGGINGFACE_API_KEY)
print("Pushed Sucessfully!")
