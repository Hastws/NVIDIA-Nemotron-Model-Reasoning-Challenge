import json, sys

with open('/tmp/v27_output/nvidia-nemotron-sfttrainer-training.log') as f:
    logs = json.load(f)

# Print last 40 entries
print('=== LAST 40 LOG ENTRIES ===')
for entry in logs[-40:]:
    stream = entry.get('stream_name', '')
    data = entry.get('data', '').strip()
    if data:
        prefix = '[ERR]' if stream == 'stderr' else '[OUT]'
        print(f'{prefix} {data[:300]}')

print(f'\n=== TOTAL ENTRIES: {len(logs)} ===')

# Find error-related entries
print('\n=== ERROR/TRACEBACK ENTRIES ===')
for i, entry in enumerate(logs):
    data = entry.get('data', '')
    if any(kw in data for kw in ['Traceback', 'Exception', 'FAILED', 'RuntimeError', 'ValueError', 'KeyError', 'assert']):
        print(f'  [{i}/{len(logs)}] {data[:400]}')
