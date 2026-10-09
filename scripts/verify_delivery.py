"""Check saved evidence and optionally refresh the final result hash manifest."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse
import json
import xml.etree.ElementTree as ET
from urllib.request import urlopen

from pypdf import PdfReader
from sna_project.utils import source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-manifest', action='store_true')
    parser.add_argument('--url', help='Optional live local demo URL')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / 'results/export_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['complete'] and manifest['completed'] == manifest['expected'] == 15
    assert manifest['source_hash'] == source_hash()
    for filename, expected in manifest['files'].items():
        assert sha256((root / filename).read_bytes()).hexdigest() == expected, filename
    for dataset in ['Cora', 'CiteSeer', 'PubMed']:
        for run in range(5):
            identity = json.loads((root / f'artifacts/{dataset}/run_{run}/identity.json').read_text())
            assert identity['source_hash'] == manifest['source_hash']
    suites = ET.parse(root / 'results/protocol_tests.xml').getroot().findall('testsuite')
    tests = sum(int(suite.attrib['tests']) for suite in suites)
    assert tests == 12
    assert all(int(suite.attrib[key]) == 0 for suite in suites for key in ['errors', 'failures', 'skipped'])
    demo = json.loads((root / 'results/demo_audit.json').read_text())
    browser = json.loads((root / 'results/browser_verification.json').read_text())
    assert demo['passed'] and demo['runs'] == 15 and demo['pairs_checked'] == 107
    assert browser['passed'] and browser['datasets'] == 3
    reader = PdfReader(root / 'output/pdf/SNA_Project_Report.pdf')
    assert len(reader.pages) == 27
    assert min(len(page.extract_text().strip()) for page in reader.pages) > 500
    literature = json.loads((root / 'docs/literature_notes.json').read_text())
    assert len(literature) == 10
    live = None
    if args.url:
        base = args.url.rstrip('/')
        with urlopen(base + '/api/summary', timeout=20) as response:
            summary = json.load(response)
        assert summary['manifest']['complete'] and summary['manifest']['completed'] == 15
        with urlopen(base + '/report', timeout=20) as response:
            assert response.status == 200 and response.read(5) == b'%PDF-'
        live = {'url': base, 'summary_and_report_passed': True}
    record = {
        'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'completed_runs': 15, 'source_hash': manifest['source_hash'],
        'protocol_tests_passed': tests, 'demo_pairs_checked': demo['pairs_checked'],
        'browser_checks_passed': browser['checks'], 'report_pages': len(reader.pages),
        'report_sha256': sha256((root / 'output/pdf/SNA_Project_Report.pdf').read_bytes()).hexdigest(),
        'literature_reviews': len(literature), 'live_demo': live,
    }
    if args.refresh_manifest:
        (root / 'results/final_verification.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        manifest['files'] = {
            file.relative_to(root).as_posix(): sha256(file.read_bytes()).hexdigest()
            for file in sorted((root / 'results').rglob('*'))
            if file.is_file() and file != path and '__pycache__' not in file.parts
        }
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
