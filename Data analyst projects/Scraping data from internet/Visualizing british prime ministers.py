# -*- coding: utf-8 -*-
"""
Spyder Editor

Analysing data gathered from Wikipedia about prime ministers
"""

# step one: Scape data from wikipedia and get rid of unneccessary data
# step two: clean your data and fill in any missing points. (use chatgpt)
# step three: Make good graph of the data gathered.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})

data = []
for row in table.find_all('tr'):
    row_data = []
    for cell in row.find_all(['td', 'th']):
        row_data.append(cell.get_text(strip=True))
    if row_data:
        data.append(row_data)

max_cols = max(len(row) for row in data)
data = [row + [''] * (max_cols - len(row)) for row in data]

df = pd.DataFrame(data[1:], columns=data[0])  # Assuming the first row is header
df.to_csv('Objects.csv', index=False)

print('CSV file has been saved.')

#print(data)
