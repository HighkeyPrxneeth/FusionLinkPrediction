"""Create a portable source/result bundle without environments or large walk corpora."""
from pathlib import Path
from hashlib import sha256
import json
import zipfile

root = Path(__file__).resolve().parents[1]
status = json.loads((root/'results/export_manifest.json').read_text())
if not status['complete']:
    raise RuntimeError('Incomplete benchmark cannot be packaged as a final submission')
if not (root/'output/pdf/SNA_Project_Report.pdf').is_file():
    raise RuntimeError('Build the final report first')
files = []
for folder in ['src','tests','configs','scripts','demo','docs','results','artifacts','output/pdf','data/processed']:
    for path in (root/folder).rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.name.endswith('.tmp'):
            continue
        if path.name == 'deepwalk_corpus.txt':
            continue
        if 'docs/sources' in path.as_posix() and not (path.name.endswith('_metadata.json') or path.name == 'retrieval_manifest.json'):
            continue
        files.append(path)
for name in ['README.md','REPORT.md','PROJECT_PLAN.md','pyproject.toml','requirements.txt','requirements-lock.txt','environment.yml','conda-explicit-win64.txt','.gitignore']:
    files.append(root/name)
manifest = {path.relative_to(root).as_posix():sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
output = root/'output/SNA_Project_Submission.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(files):
        archive.write(path,'SNA-Project/'+path.relative_to(root).as_posix())
    archive.writestr('SNA-Project/SUBMISSION_MANIFEST.json',json.dumps({'files':manifest,'completed_runs':15,'excluded':['conda environment','downloaded raw data','walk corpora','archived paper full texts']},indent=2))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(files)+1
print(f'Created {output}: {output.stat().st_size/1e6:.1f} MB; {len(files)} files verified')
