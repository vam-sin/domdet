import requests
import json
"F4HVG8"
url = "https://alphafold.ebi.ac.uk/files/AF-F4HVG8-F1-predicted_aligned_error_v2.json"

url2 = "https://alphafold.ebi.ac.uk/files/AF-P03023-F1-predicted_aligned_error_v2.json"

def

r = requests.get(url)
print(r.json())
bp=True
