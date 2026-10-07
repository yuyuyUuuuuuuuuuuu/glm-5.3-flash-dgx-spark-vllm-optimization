"""Host cost of exact_lens per metadata build (CPU only): decode-only steps (the common case) and a mixed step."""
import sys, time, types
sys.path.insert(0, "/w")
import torch
import glm53_mla_exactlens as X
fake = types.SimpleNamespace(_index_topk=2048, _index_kpool=4, vllm_config=types.SimpleNamespace(
    speculative_config=types.SimpleNamespace(num_speculative_tokens=7)))
def cam(reqs):
    ql = torch.tensor([q for _, q in reqs], dtype=torch.int32)
    qsl = torch.zeros(len(reqs) + 1, dtype=torch.int32); qsl[1:] = torch.cumsum(ql, 0)
    return types.SimpleNamespace(num_reqs=len(reqs), num_actual_tokens=int(qsl[-1]), max_query_len=int(ql.max()),
                                 query_start_loc_cpu=qsl, seq_lens_cpu_upper_bound=torch.tensor([s for s, _ in reqs], dtype=torch.int32))
for name, reqs in (("1 req x 8 rows (decode)", [(50000, 8)]), ("8 req x 8 rows (decode)", [(50000 + i, 8) for i in range(8)]),
                   ("4 decode + 1 prefill chunk", [(9000 + i, 6) for i in range(4)] + [(30000, 13824)])):
    c = cam(reqs)
    n = c.num_actual_tokens
    ctx = torch.cat([torch.arange(s - q + 1, s + 1) for s, q in reqs])
    lens = torch.where(ctx <= 2048, ctx, 2048 + ctx % 4).to(torch.int32)
    for _ in range(200):
        X.exact_lens(fake, c, n, lens)
    t0 = time.perf_counter(); N = 2000
    for _ in range(N):
        X.exact_lens(fake, c, n, lens)
    print(f"{name}: exact_lens {1e6 * (time.perf_counter() - t0) / N:.1f} us per build")
