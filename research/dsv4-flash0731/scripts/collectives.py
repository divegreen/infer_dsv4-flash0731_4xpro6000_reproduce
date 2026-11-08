"""Small, finite NCCL measurements; payload bandwidth is not physical bus bandwidth."""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
    rows = []
    for size in (8192, 32768, 262144, 1048576, 16777216):
        elements = size // 4
        for operation in ('all_reduce', 'all_gather', 'all_to_all'):
            source = torch.ones(elements, device=f'cuda:{rank}', dtype=torch.float32) * rank
            target = torch.empty(elements * world if operation == 'all_gather' else elements, device=source.device, dtype=source.dtype)
            if operation == 'all_reduce':

                def call():
                    dist.all_reduce(source)
            elif operation == 'all_gather':

                def call():
                    dist.all_gather_into_tensor(target, source)
            else:

                def call():
                    dist.all_to_all_single(target, source)
            call()
            torch.cuda.synchronize()
            if operation == 'all_reduce':
                assert torch.all(source == sum(range(world))).item()
            else:
                blocks = target.reshape(world, -1)
                assert all((torch.all(blocks[i] == i).item() for i in range(world)))
            source.zero_()
            for _ in range(10):
                call()
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                dist.barrier()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for _ in range(50):
                    call()
                torch.cuda.synchronize()
                elapsed = torch.tensor((time.perf_counter() - begin) / 50, device=source.device, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
            if rank == 0:
                row = {'world_size': world, 'operation': operation, 'input_bytes_per_rank': size, 'iterations_per_trial': 50, 'trials_seconds_per_call': times, 'median_us': statistics.median(times) * 1000000.0, 'input_payload_GBps': size / statistics.median(times) / 1000000000.0}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if rank == 0:
        output = Path(os.environ['RESEARCH_RUN_DIR']) / 'collectives.json'
        write_text(output, json.dumps({'torch': torch.__version__, 'nccl': torch.cuda.nccl.version(), 'metrics': rows}, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
