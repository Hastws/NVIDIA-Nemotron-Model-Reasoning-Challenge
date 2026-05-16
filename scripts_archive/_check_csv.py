import csv, sys
for f in sys.argv[1:]:
    with open(f) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        print(f"{len(rows)} records, cols={rows[0].keys() if rows else 'empty'} — {f.split('/')[-1]}")
        if rows:
            r = rows[0]
            print(f"  thinking len={len(r.get('thinking',''))}, answer={r['answer'][:50]}")
