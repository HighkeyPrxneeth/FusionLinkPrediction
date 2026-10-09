"""Cache public primary-source documents and Crossref metadata without credentials."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'sources'
SOURCES = {
    'modularity': ('10.1016/j.neunet.2022.06.021', 'https://arxiv.org/pdf/2202.00961'),
    'experimental_survey': ('10.1007/s41019-022-00188-2', 'https://link.springer.com/article/10.1007/s41019-022-00188-2'),
    'stacking_critique': ('10.1016/j.isci.2022.105697', 'https://www.cell.com/iscience/fulltext/S2589-0042(22)01970-8'),
    'gaepnas': ('10.3390/e25040567', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10138145/fullTextXML'),
    'fair_benchmark': ('10.3390/a15100379', 'https://pdfs.semanticscholar.org/ad79/1a279e7801e29de3968a4d752ee0f3c4161e.pdf'),
    'mcogcn': ('10.1038/s41598-024-80509-9', 'https://www.nature.com/articles/s41598-024-80509-9.pdf'),
    'sequential_stacking': ('10.1038/s41467-024-45598-0', 'https://www.nature.com/articles/s41467-024-45598-0.pdf'),
    'se4lp': ('10.1007/s11036-024-02339-3', 'https://link.springer.com/article/10.1007/s11036-024-02339-3'),
    'tagnn': ('10.1038/s41598-026-48184-0', 'https://www.nature.com/articles/s41598-026-48184-0.pdf'),
    'lpgnn': ('10.1109/ACCESS.2026.3677526', 'https://shura.shu.ac.uk/37825/1/Tangirala-LinkPredictionUsing%28VoR%29.pdf'),
    'dynamic_gcn': ('10.1038/s41598-023-50977-6', 'https://www.nature.com/articles/s41598-023-50977-6.pdf'),
    'high_frequency': ('10.1038/s41598-024-56144-9', 'https://www.nature.com/articles/s41598-024-56144-9.pdf'),
    'ncsm': ('10.1016/j.neucom.2024.127820', 'https://www.sciencedirect.com/science/article/pii/S0925231224005915'),
}


def fetch(key, doi, url):
    result = {'key': key, 'doi': doi, 'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    endpoints = [('metadata', 'https://api.crossref.org/works/' + doi), ('document', url)]
    for kind, endpoint in endpoints:
        try:
            existing = sorted(OUT.glob(key + '_' + kind + '.*'), key=lambda p: (p.suffix != '.pdf', p.name))
            if existing and (kind == 'metadata' or existing[0].suffix == '.pdf' or not endpoint.endswith('.pdf')):
                path = existing[0]
                data = path.read_bytes()
                result[kind] = {'file': str(path.relative_to(ROOT)), 'sha256': sha256(data).hexdigest(), 'bytes': len(data), 'content_type': 'cached'}
                continue
            req = urllib.request.Request(endpoint, headers={'User-Agent': 'SNA-Coursework/1.0 (academic literature provenance)'})
            with urllib.request.urlopen(req, timeout=40) as response:
                data = response.read()
                content_type = response.headers.get('Content-Type', '')
            suffix = '.json' if kind == 'metadata' else ('.pdf' if data.startswith(b'%PDF') else '.html')
            path = OUT / (key + '_' + kind + suffix)
            path.write_bytes(data)
            result[kind] = {'file': str(path.relative_to(ROOT)), 'sha256': sha256(data).hexdigest(), 'bytes': len(data), 'content_type': content_type}
        except Exception as error:
            result[kind] = {'error': str(error)}
    return result


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, key, doi, url) for key, (doi, url) in SOURCES.items()]
        for job in as_completed(jobs):
            record = job.result()
            print(record['key'], {k: ('saved' if 'file' in record[k] else record[k]['error']) for k in ['metadata', 'document']}, flush=True)
            results.append(record)
    (OUT / 'retrieval_manifest.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
