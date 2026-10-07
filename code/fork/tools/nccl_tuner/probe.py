import os, torch, torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29544")
dist.init_process_group("nccl", rank=0, world_size=1)
for n in (64 * 1024, 4 * 2**20):
    x = torch.ones(n // 2, dtype=torch.bfloat16, device="cuda"); dist.all_reduce(x); torch.cuda.synchronize()
print("PROBE_OK")
dist.destroy_process_group()
