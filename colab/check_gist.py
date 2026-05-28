import json
import os

p = os.path.join(os.environ['TEMP'], 'gist_check.json')
nb = json.load(open(p))
print(f'Cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    print(f'Cell {i}: src_len={len(c["source"])}')
    if i == 1:
        print('---CONTENT---')
        print(''.join(c['source']))
        print('---END---')
