#!/usr/bin/env python3
"""CPU (torch, no GPU) semantics of GLM53_KPOOL_TAIL_POSITIONS=2 on the patched KpoolTailMetadataBuilder.build:
the circular slots are written INTO the persistent slot-mapping buffer (same data_ptr as the input), padding tokens
keep PAD (-1), and value 1's form returns a NEW tensor (the address a FULL CUDA graph captured is not the one written).
Usage (inside the production image, CPU only): python3 test_tail_inplace_cpu.py <patched indexer.py> <pristine indexer.py>"""
import ast, sys, types
import torch

def load_fn(path):
    src = open(path).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_kpool_tail_slot_mapping")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KpoolTailMetadataBuilder")
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    # body of build up to the slot_mapping decision, run as a function returning slot_mapping
    body = []
    for st in build.body:
        if isinstance(st, ast.Return):
            break
        body.append(st)
    f = ast.FunctionDef(name="b", args=ast.arguments(posonlyargs=[], args=[ast.arg("self"), ast.arg("common_attn_metadata")],
                        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body + [ast.Return(ast.Name("slot_mapping", ast.Load()))],
                        decorator_list=[], returns=None, type_params=[])
    ast.fix_missing_locations(f)
    ns["split_decodes_and_prefills"] = lambda c: (0, 0, 0, 0)
    exec(compile(ast.Module([f], []), path, "exec"), ns)
    return ns["b"]

FAIL = []
def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c: FAIL.append(m)

kpool = 16
self = types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=kpool))
# FULL-graph-like batch: 2 real requests x 8 spec-verify tokens, padded to 3 reqs / 24 tokens
persistent = torch.full((32,), -1, dtype=torch.int64)
n_pad = 24
qsl = torch.tensor([0, 8, 16, 16], dtype=torch.int32)          # padded req has qsl == total real tokens
pos = torch.zeros(32, dtype=torch.int64)
pos[:8] = torch.arange(1000, 1008); pos[8:16] = torch.arange(2003, 2011); pos[16:24] = 777  # padding garbage
bt = torch.zeros(3, 4, dtype=torch.int32); bt[0, 0] = 5; bt[1, 0] = 9; bt[2, 0] = 3; bt[0, 1:] = 42  # stale cols
persistent[:16] = 0      # generic kernel output: zero column -> block 0
cam = types.SimpleNamespace(slot_mapping=persistent[:n_pad], positions=pos[:n_pad], block_table_tensor=bt,
                            query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), num_actual_tokens=n_pad, num_reqs=3,
                            seq_lens=None, max_seq_len=0)
b2 = load_fn(sys.argv[1]); b1 = load_fn(sys.argv[2])
ptr = persistent.data_ptr()
out1 = b1(self, cam)
check(out1.data_ptr() != ptr, "value 1 (pristine builder + positions) returns a NEW tensor (FULL replay never reads it)")
check(int((out1[16:24] >= 0).sum()) == 8, "value 1 maps the 8 FULL-padding tokens into a real ring (pads lose PAD_SLOT_ID)")
before = persistent.clone()
out2 = b2(self, cam)
check(out2.data_ptr() == ptr, "value 2 returns the persistent buffer itself")
exp = torch.cat([5 * kpool + torch.arange(1000, 1008) % kpool, 9 * kpool + torch.arange(2003, 2011) % kpool])
check(torch.equal(persistent[:16], exp), f"value 2 writes own_block*16 + pos%16 in place for real tokens: {persistent[:16].tolist()}")
check(torch.equal(persistent[16:], before[16:]), "value 2 leaves padding (PAD -1) untouched")
# piecewise-like call (no padding): num_reqs == real
persistent.fill_(0)
cam2 = types.SimpleNamespace(**{**vars(cam), "num_actual_tokens": 16, "num_reqs": 2, "slot_mapping": persistent[:16],
                                "positions": pos[:16], "query_start_loc": qsl[:3], "query_start_loc_cpu": qsl[:3].clone()})
b2(self, cam2)
check(torch.equal(persistent[:16], exp), "value 2, unpadded batch: same circular slots")
print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
