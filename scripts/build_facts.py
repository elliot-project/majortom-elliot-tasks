"""Build the facts store: one record per tile, `factstore.SHARD` tiles per file.

    ELLIOT_ROOT=... ELLIOT_X_EXT_ROOT=... python scripts/build_facts.py OUT [--workers N]
        [--parts monthly,burst,monotemporal]

Workers claim shards by creating `<OUT>/<part>/facts-<first>.claim` exclusively, so
any number of these processes, on any number of machines, can share one OUT; a shard
whose parquet exists is done. A tile whose facts cannot be computed is written with
`facts` null and its error, never skipped. Delete stale `.claim` files to retry a
shard whose worker died.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def shards(out: Path, parts: list[str]) -> list[tuple[str, int, int]]:
    import elliot_tasks as et
    from elliot_tasks import factstore as F
    todo = []
    for part in parts:
        n = len(et.open_part(part))
        todo += [(part, s, min(s + F.SHARD, n)) for s in range(0, n, F.SHARD)]
    return todo


def run_shard(out: Path, part: str, start: int, stop: int) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import elliot_tasks as et
    from elliot_tasks import factstore as F
    final = F.shard_path(out, part, start)
    if final.exists():
        return f'{final.name} done already'
    claim = final.with_suffix('.claim')
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return f'{final.name} claimed elsewhere'
    os.write(fd, f'{socket.gethostname()} {os.getpid()} {time.ctime()}\n'.encode())
    os.close(fd)
    t0 = time.time()
    rows = {'index': [], 'cell': [], 'facts': [], 'error': []}
    for i in range(start, stop):
        tile = et.tile(part, i)
        try:
            rec = F.compute(tile)
            rows['facts'].append(F.encode(rec))
            rows['error'].append(None)
            cell = rec['cell']
        except Exception:
            rows['facts'].append(None)
            rows['error'].append(traceback.format_exc(limit=3))
            cell = tile.cell
        rows['index'].append(i)
        rows['cell'].append(cell)
    table = pa.table({'index': pa.array(rows['index'], pa.int32()),
                      'cell': pa.array(rows['cell'], pa.string()),
                      'facts': pa.array(rows['facts'], pa.large_string()),
                      'error': pa.array(rows['error'], pa.string())})
    tmp = final.with_suffix(f'.{os.getpid()}.tmp')
    pq.write_table(table, tmp, compression='zstd', compression_level=9)
    os.replace(tmp, final)
    claim.unlink(missing_ok=True)
    errors = sum(e is not None for e in rows['error'])
    return f'{final.name} {stop - start} tiles, {errors} errors, {time.time() - t0:.0f}s'


def _init():
    import warnings
    warnings.filterwarnings('ignore')
    import elliot_tasks as et
    et.configure()


def _work(args):
    out, part, start, stop = args
    try:
        return f'{part} {run_shard(Path(out), part, start, stop)}'
    except Exception:
        return f'{part} {start} FAILED\n{traceback.format_exc()}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--parts', default='monthly,burst,monotemporal')
    a = ap.parse_args()
    out = Path(a.out)
    parts = a.parts.split(',')
    for p in parts:
        (out / p).mkdir(parents=True, exist_ok=True)
    _init()
    todo = [(str(out), *s) for s in shards(out, parts)]
    print(f'{len(todo)} shards, {a.workers} workers on {socket.gethostname()}', flush=True)
    failed = 0
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, maxtasksperchild=4) as pool:
        for msg in pool.imap_unordered(_work, todo):
            failed += 'FAILED' in msg
            print(time.strftime('%H:%M:%S'), msg, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
