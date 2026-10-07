# What did not work

Every idea here was measured in production (2 x GB10, TP=2) unless it is marked "node C", which was a
single GB10 used for kernel tests. "Rejected" means it was measured and not adopted. "Reverted" means it
was deployed and then rolled back. Dates are 2026. Decode numbers are from `bench_decode.py` (n=10,
temperature 0, idle) unless stated otherwise.

## Formats and checkpoints

| idea | date | measured result | why it failed |
|---|---|---|---|
| NVFP4 (W4A16 compressed-tensors) as the serving format | 09-02 | decode about 30 tok/s against 47-52 for EXL3. Weights 192 GB against 165 GB. Host MemAvailable about 2 GB. Four OOM freezes | slower and no memory headroom |
| NVFP4 swap on the live cluster (KV left at 15 GiB) | 09-13 | weights 82.37 -> 90.89 GiB per node, KV unchanged. Both nodes ran out of memory *after* the health check passed | "models endpoint = 200" is not capacity proof. Reduce KV when weights grow |
| NVFP4 GEMM for decode | 09-23 | `cutlass_scaled_fp4_mm_sm120a` runs only for m = 128-640 in steps of 128. m = 1..96 and 192 fail with "Error Internal" (numerics are correct where it runs, 276-283 TFLOP/s for prefill shapes) | decode shapes (m = 1-8) cannot use it. `supports_fp4(121)=True` proves nothing |
| switching to NVFP4 to reach 3,000 tok/s prefill (as a public GX10 report did) | 09-30 | KV pool 2.0M -> about 1.1M tokens | loses the 1M context. Even NVFP4 with a 1M-class KV would not reach 3,000 |
| NVFP4 MLA KV for a 3M+ pool | 09-12 | KLD 0.024555 -> 0.060535 (2.47x worse; 25 windows / 51,175 positions) | quality |
| dropping the all-zero RoPE region of MLA KV (656 -> 528 B/token/layer) | 09-12 | not implementable | 656 is a literal in the sparse-MLA kernel with an FFI size check. The RoPE offset is read unconditionally |

## Scheduling and memory knobs

| idea | date | measured result | why it failed |
|---|---|---|---|
| MNBT 7168 -> 16384 (first try) | 09-02 | 100K prefill 1,184 -> 889 tok/s, short-prompt TTFT 0.49 -> 13.1 s | measured with that day's recipe settings (mixed-prefill and sequence settings not recorded). On 09-07, with 8 sequences, a 512k window and the right-sized indexer workspace, 16384 won and was adopted. The cause of the difference was not isolated |
| MNBT 32768 | 09-07 | 666 tok/s at 16k, then CUDA illegal memory access at about 40k | the limit lies between 16384 and 32768, not at 8192 |
| `MIXED_PREFILL_CHUNK=skip` ("optimal" on 09-06) | 09-07 | interactive request during 3 heavy jobs: 83.4 s (skip) against 0.76 s (0) | the sweep measured only running decoders |
| `MIXED_PREFILL_CHUNK=0` side effect (still in production) | 09-24 | 38% of 10-s windows below 1 tok/s. Steps of 10-40 s were 0.63% of steps but 37% of decode time | a 16k chunk at 300k+ context takes about 20 s and joins every step. Mitigated by APC retention fixes, not by changing the chunk |
| long-prefill token threshold 2048 | 09-24 | would cut single-stream prefill 1,203 -> 882 tok/s | withdrawn before deployment |
| KV sized for exactly 1M tokens (7.5 GiB) | 09-30 | did not start. A 1M-token request needs 8.18 GiB, so the minimum is 8.25 GiB (1,006,289 tokens) | arithmetic error, about 10 min of downtime |
| KV 8.25 GiB + MNBT 32256 (bigger prefill chunks) | 09-30 | 32k gate 1,849-1,851 against 1,846-1,853. Minimum MemAvailable 1.73 against 4.0 GiB | no speed-up. The chunk's working set costs more than the KV saved |
| `NCCL_NCHANNELS=4` | 09-28 | decode -0.4%, 24k prefill -3% | large prefill all-reduces need the channels |
| forced NCCL Simple-protocol tuner plugin | 09-28 | all-reduce 8.4 -> 11.4 ms | slower |
| SM120 sparse-MLA backend (default selection on cc 12.1) | 09-04 / 09-06 | 15,591-23,201 B/token against 7,227 for SM90. The pool shrank to 59% after a rebuild silently dropped the patch | the patch must be kept. Rebuilds without "skip build" lose it |
| restarting only the head rank | 09-09 | worker kept a broken NCCL with 260 CLOSE-WAIT sockets while `docker ps` showed it healthy | restart both ranks |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` (from a wrong "cache is broken" diagnosis) | 09-12 | the test prompt (4,022 tokens) was below the 4,608 block size. Real traffic had 98-99% hits all along | removed |
| kpool indexer: drop the lowest-scored pool deterministically | 10-01 | decode-vs-prefill KL 0.00749 -> 0.01090 (+45%), prefill -1.7% | worse consistency, reason not established |

## Speculative decoding

| idea | date | measured result | why it failed |
|---|---|---|---|
| fixed K=3 | 09-13 | prose +31%, JSON -29% (k=7: prose 15.5 / JSON 52.5; k=3: 20.3 / 37.2 tok/s) | no single K wins. Adaptive K {4,5,7} solved it on 09-22 |
| adaptive K sets {4,7}, {3,4,7}, {4,5,6,7} | 09-22 | sums 137.79 / 138.39 / -1.7% against 144.52 | the gain-against-step-cost trade-off differs by phase |
| EMA alpha / margin 0.5 | 09-22 | +0.9% / -2.2% | noise |
| sharing token 0's top-8 experts across all 8 speculative positions | 09-13 | prose 16.6 -> 28.9, code 35.1 -> 88.6 tok/s, but 12-question set 7/8 -> 5/8, inputs echoed back, invented glyphs, JSON ignored | accepted positions keep the wrong experts' output. Verification does not protect quality here |
| MTP (k=3) instead of DFlash2 | 09-13 | prose 17.3-18.3, JSON 34 tok/s | below DFlash2 k=3 (20.3) |
| self-made FP8-MLP DFlash2 draft | 09-13 | prose 7.2 / JSON 7.1 against bf16 16.5 / 50.1 tok/s | acceptance collapsed |
| retrained third-party DFlash2 drafter ("+68%" claim) | 09-22 | prose -6.1%, structured -5.7%. At K=2, structured 71.66 -> 29.18 | the claim was at TP=4, K=2, temp 0, with a baseline of 1.07 accepted/step. Ours was already 2.02-6.71 |
| DSpark preview drafter | 09-29 | T=1.0: prose 30.5 against 42.2 tok/s (accepted 2.89 against 3.18), coding 42.1 against 55.9, structured 69.2 against 89.8. Tuned: -15 to -20% | lower acceptance on our traffic. Needed fp8 drafter KV to fit the KDA page. 15 min of downtime over two failed starts |
| bf16 drafter instead of FP8 (offline equivalence said FP8 top-1 was only 0.75-0.82) | 09-30 | acceptance unchanged (prose 3.17 -> 3.11), step +3.5 ms, prose 42.3 -> 39.9 tok/s | a synthetic-input mismatch did not show up in real-traffic acceptance |
| drafter KV in fp8 | 09-13 | +112k pool tokens, no speed change | rolled back with the rest of that session. Later required by DSpark only |

## Kernels and decode overlays

| idea | date | measured result | why it failed |
|---|---|---|---|
| thin-decode MoE (upstream opt-in) | 09-22 | +0.1-2.7%, p > 0.74. Upstream also reports a bimodal 15-30% drop in 1 of 4 runs | not significant |
| cooperative decode MoE overlay (upstream) | 09-22 | sum -8.4% (prose -16.6%, coding -13.8%), ms/step 105 -> 111-122. Build digest not reproducible | slower |
| InstantTensor loading alone | 09-22 | prose -7.9% in isolation | kept only as part of the upstream stack |
| MoE SMs per expert 8 -> 4 | 09-13 | no change (15.9 / 31.8 / 56.1 tok/s) | no gain measured. The routed MoE kernel was already at its bandwidth floor at k=7 (`decode-anatomy.md`) |
| torch.compile (Dynamo enabled by turning off breakable cudagraphs) | 09-13 | 16.9 / 49.6 tok/s, unchanged | no gain measured |
| drafter TP=1 | 09-13 | no change | no gain measured. Collectives were about 3.8% of the step, so even the predicted effect was only 1.02-1.07x |
| unfused EXL3 MoE path | 09-13 | failed to start | not investigated further; rolled back with that session |
| "reconstruct EXL3 experts to BF16 for decode" | 09-18 | not a switch. The log line is a hard-coded string, and BF16 would need about 4x the memory | closed by reading the source |
| L2 pre-read of the next weights during decode (FP8 GEMV / MoE glue warm-up) | 09-29 | -5.8 ms in isolation. In production: all-reduce p50 31 -> 50 us (98 us with warm-up), mHC 13 -> 22 us, router 16 -> 27 us. Prose 72.4-72.8 against 73.6 ms = no gain. With CTA 8 -> 2: 81.9 ms | GB10 shares DRAM between GPU and NIC, so there is no free window. Node C has no real RoCE all-reduce to show it |
| side-stream fork/join inside decode graphs | 09-29 | boot failure in production ("capturing stream has unjoined work"), about 14 min down | vLLM uses breakable piecewise CUDA graphs for this model. Node C tests captured whole graphs only |
| KDA lazy state write-back v1 | 10-04 | decode-vs-prefill KL 0.0115 -> 0.206, coding acceptance 4.13 -> 2.73, temperature-0 outputs differed every run, Japanese garbled. The structured benchmark alone looked fine | layers were keyed by state pointer only. v2 keys by (state ptr, A_log ptr) and was adopted |
| migrating the server to TensorFold | 09-29 | its official 2x GB10 numbers on the same checkpoint: 29.7-43.8 tok/s, below production. 4-stream structured 146.5 (vLLM) against about 78. No logprobs, no n>1 | about 30 production fixes would need rebuilding to gain nothing |
| NInfer GB10 port | 09-23 | Qwen-only, single GPU, no TP | cannot host a 320B TP=2 model |
| faster all-reduce backends | 09-12 | NCCL symmetric memory needs a world size of at least 4. Quick-reduce is ROCm-only. Custom all-reduce needs NVLink. Symmetric-memory all-reduce needs GPUDirect RDMA, which GB10 lacks | world size 2. Collectives were only about 6.4 ms (3.8%) of the step anyway |

## Prefill precision arms (32k gate, idle)

These are rejected arms from the 10-02 to 10-04 A/B runs. The acceptance limits were long KL <= 0.0165,
dvp <= 0.023 and top-1 >= 98.3. "base" is the production configuration measured in the same run.

| arm | gate32k (tok/s) | long KL | dvp KL | reason |
|---|---|---|---|---|
| W8A8 on all dense projections | 2,751.5 | 0.0154 | 0.0242 | dvp over the limit |
| W8A8 all except mla.o_proj | 3,122.5 | 0.0241 | 0.0195 | long KL |
| W8A8 all except mla.o_proj on layers 19-39 ("a2") | 3,134.5 / 3,170 | 0.0164 / 0.0169 | 0.0153 / 0.0172 | NLL +0.0012, significantly worse |
| W8A8 hi+lo (two boots) | 3,182 / 3,186 | 0.0231 / 0.0232 | 0.0223 / 0.0183 | long KL |
| mHC fused + round-A | 2,998 | 0.0176 | 0.0184 | long KL |
| sequence-parallel mHC v2 (odd-T tail) | 3,042 (base 3,016.5) | 0.0153 | 0.0212 | +0.8%, below the 1-2% margin |
| e4m3 mainloop alone | 2,728 (base 2,690) | 0.0140 | 0.0176 | +1.4%, below margin (adopted later as part of a bundle) |
| e4m3 off, exact fp16 MoE | 2,437.5 (base 3,027.5) | 0.0065 | 0.0103 | best quality (NLL -0.0020) but -19.5% speed |
| fp32 accumulate (no bf16 acc / fold) | 2,805.5 | 0.0157 | 0.0185 | -7.3%, no NLL benefit |
| exact first 4 / first 8 / last 8 / last 4 MoE layers | 2,952.5 / 2,894.5 / 2,895 / 2,953-2,959.5 | 0.013-0.014 | 0.017-0.019 | -2.5 to -4.4% for at most about 0.001 NLL (often not significant) |

## Conclusions we retracted

These are recorded because the reasoning that produced them is common:

- **"Effective concurrency 1"** (09-05). It was measured while other requests were running, with
  thinking on and `max_tokens=400` all spent on reasoning. On an idle engine, JSON throughput scales to
  131.4 tok/s at N=8.
- **"Prefix cache is broken"**, twice. In both cases the test prompt was shorter than the cache block
  (2,144 tokens on another model, 4,608 here).
- **"cutlass_80 (an Ampere kernel name) is slow on sm_121"**. That bf16 GEMV runs at 246.9-249.2 GB/s,
  90% of spec.
- **"Decode is bandwidth-bound"**, then **"the MoE kernel is 7x off its floor"**. At k=7 the routed MoE
  runs at 0.98x its floor (1,434 us against 1,464 us). With speculation off it is latency-bound.
- **"The 105 ms step = verifier 55.4 + drafter 49.6 ms"**. The drafter runs once per step. A larger K
  grows the verify MoE (more distinct experts).
- **"The 2M KV pool is impossible"**. It was computed with the old bytes per token.
- **"Prose +10.7%"** at n=3. It was +0.1% (p=0.998) at n=10.
- **"Prefix cache 17.7x"**. In seconds it was a 2.5x cold regression.
- **"e4m3 loss is concentrated in the first layers"**. This came from a single-node mini rig. In
  production it is the deep layers.
- **"Sorting kpool by score worsens dvp"**. The mini rig compared different greedy sequences per arm.
- **The first kernel-fork build "works"**. It never ran on the production path: weights were not wired,
  and the SiLU flag and dtype assumptions were wrong. An early "1.37x" compared against the prefill
  kernel, which is the wrong baseline.
- **A first task-benchmark score of 0.1997 against 0.4579 ("GLM loses")**. 32 of 33 trials hit the
  90-minute cap on a broken harness.
