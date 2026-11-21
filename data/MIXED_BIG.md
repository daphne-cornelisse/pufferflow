# mixed_big data

`config/mixed_big.ini` trains on `data/mixed_big`: 10,007 maps. A clone already has the I24 logs and the four CARLA towns. The Waymo training set is the download.

| Maps | Source |
| --- | --- |
| 0–9999 | Waymo Open Motion training scenes |
| 10000–10002 | I24 scenes 1, 2, and 3 (`data/i24/binaries`) |
| 10003–10006 | Town01, Town02, Town06, Town10HD (`data/mixed/binaries`) |

Town03, Town04, Town05, and Town07 are not in the pool. Their roads change height, and this build has no 3D roads. The four town files in the repo are the CARLA maps to use.

## Waymo

The archive is `training.zip` (4.5 GB) from [daphne-cornelisse/pufferdrive_womd_train](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train). The JSON inside expands to about 19 GB.

```bash
pip install huggingface_hub tqdm
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='daphne-cornelisse/pufferdrive_womd_train', repo_type='dataset', filename='training.zip', local_dir='data/womd_train_10k')"
mkdir -p data/womd_train_10k/json
unzip -j data/womd_train_10k/training.zip 'data/processed/training/*.json' -d data/womd_train_10k/json
python utils/womd.py --data_dir data/womd_train_10k
```

That writes `data/womd_train_10k/binaries/map_000.bin` through `map_9999.bin`, plus `sdc.bin`. Map order is the sorted JSON names. This is the 10,000-scene training split, not `pufferdrive_womd_train_1000` and not the validation zip.

## Assemble

Hardlinks keep a second copy off the disk, so `data/womd_train_10k` and `data/mixed_big` have to be on the same filesystem. From the repo root:

```bash
python - << 'PY'
import os
import struct
from pathlib import Path

root = Path(".").resolve()
out = root / "data" / "mixed_big" / "binaries"
out.mkdir(parents=True, exist_ok=True)

def link(src, name):
    dst = out / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)

womd = root / "data" / "womd_train_10k" / "binaries"
for i in range(10000):
    link(womd / f"map_{i:03d}.bin", f"map_{i:03d}.bin")
for j, i in enumerate(range(10000, 10003)):
    link(root / "data" / "i24" / "binaries" / f"map_{j:03d}.bin", f"map_{i:03d}.bin")
for j, i in enumerate(range(10003, 10007)):
    link(root / "data" / "mixed" / "binaries" / f"map_{j:03d}.bin", f"map_{i:03d}.bin")

raw = (womd / "sdc.bin").read_bytes()
n = struct.unpack_from("i", raw, 0)[0]
assert n == 10000, n
idx = list(struct.unpack_from(f"{n}i", raw, 4))
idx.extend([0, 0, 0, -1, -1, -1, -1])
blob = struct.pack("i", len(idx)) + struct.pack(f"{len(idx)}i", *idx)
(out / "sdc.bin").write_bytes(blob)
print(f"{len(list(out.glob('map_*.bin')))} maps, sdc {len(blob)} bytes")
PY
```

`sdc.bin` is one int count, then one int32 self-driving-car index per map. The four towns are `-1` (no log). The three I24 maps are `0`. Expect `10007 maps, sdc 40032 bytes`.

The I24 binaries in the clone are one log per scene. The focal tracks are scene 1 id 221, scene 2 id 93, and scene 3 id 148. Regenerating them with `--all-sdcs` writes a different catalog. The raw BMVC files, if you need them, are under `data/i24/raw`, and `python utils/i24.py --scene N` rebuilds one scene.
