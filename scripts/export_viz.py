from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--out-dir',default='testOut'); a=p.parse_args()
    run=Path(a.run_dir); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in ['main.html','style.css','vis.js']:
        (out/f).write_text((Path('viz')/f).read_text())
    traj=np.load(run/'trajectory_best.npz')['observations']
    summary={'steps':int(traj.shape[0]),'obs_dim':int(traj.shape[1]) if traj.ndim>1 else 0,'eval':json.loads((run/'eval_summary.json').read_text())}
    (out/'run_data.json').write_text(json.dumps({'summary':summary}))
    print('export_viz_done', out)
if __name__=='__main__':
    main()
