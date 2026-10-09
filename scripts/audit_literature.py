from pathlib import Path
from hashlib import sha256
import json

root = Path(__file__).resolve().parents[1]
notes = json.loads((root/'docs/literature_notes.json').read_text())
manifest = {r['key']:r for r in json.loads((root/'docs/sources/retrieval_manifest.json').read_text())}
assert len(notes) == len({n['key'] for n in notes}) == 10
for note in notes:
    record = manifest[note['key']]
    for kind in ['metadata','document']:
        path = root / record[kind]['file']
        assert sha256(path.read_bytes()).hexdigest() == record[kind]['sha256']
    metadata = json.loads((root/record['metadata']['file']).read_text())['message']
    assert metadata['type'] == 'journal-article'
    assert metadata['published']['date-parts'][0][0] >= 2022
    assert metadata['DOI'].lower() == record['doi'].lower()
    assert all(note[k].strip() for k in ['method','datasets','metrics','advantage','limitation','evidence'])
print('Ten distinct 2022-onwards journal publications verified; full texts, metadata and review fields present.')
