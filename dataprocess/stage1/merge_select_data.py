import sys
sys.path.append('../..')
from util.util import load_jsonl, save_jsonl

input_files = ['../../data/select/select-data-difficult.jsonl', '../../data/select/select-data-easy.jsonl']
output_file = '../../data/select/select-data-merged.jsonl'

all_data = []
for input_file in input_files:
    data = load_jsonl(input_file)
    all_data.extend(data)
    print(f'Loaded {len(data)} records from {input_file}')

print(f'Total records: {len(all_data)}')

save_jsonl(all_data, output_file)
print(f'Saved merged data to {output_file}')
