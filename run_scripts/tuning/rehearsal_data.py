"""Build three-machine fixtures from existing smoke evidence; never train."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from run_scripts.tuning.collect import main as collect

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('screen','confirm'),required=True)
    a=p.parse_args()
    source=Path(__file__).with_name('smoke_evidence.zip')
    with zipfile.ZipFile(source) as z:
        attempts={n.rsplit('/',1)[0]:json.loads(z.read(n)) for n in z.namelist() if n.endswith('/identity.json')}
        for role in 'abc':
            root=a.root/role
            for prefix,ident in attempts.items():
                if ident['machine_role']!=role or a.stage=='screen' and ident['seed']!=101:
                    continue
                for name in z.namelist():
                    if name.startswith(prefix+'/'):
                        dest=root/name
                        if len(str(dest.resolve()))>=240: raise ValueError('fixture path too long')
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        dest.write_bytes(z.read(name))
            out=a.root/'z'/f'tuning_{role}_{a.stage}.zip'
            out.parent.mkdir(parents=True,exist_ok=True)
            collect(['--results-root',str(root),'--mode','smoke','--out',str(out)])
            Path(str(out)+'.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')

if __name__=='__main__': main()
