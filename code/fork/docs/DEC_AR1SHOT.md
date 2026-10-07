# DEC_AR1SHOT — the decode all-reduce of the 2-rank TP pair in one network hop (`GLM53_DEC_AR1SHOT`)

Branch `decode6-lat` (from `decode4-kdalazyfix` c61e3e1, production), 2026-10-05. nodeC only (one GB10, production
image, every GPU job under `flock /tmp/tf-gpu-bench.lock` through `tests/gpu_run.sh`); production read-only (container
env via `docker inspect`, log grep). Logs: `${HOME}/tf-exl3-assets/decode6/logs/`, analysis scripts
`${HOME}/tf-exl3-assets/decode6/ana/`.

## 0. Result in one paragraph

Every decode step runs **102 NCCL all-reduces** (R15 rank-0 trace `decode-prof/p6h.json.gz`: 90 in the target graph,
10 in the drafter graph, 2 eager embedding all-reduces; [M, 4096] bf16 = 40-64 KiB, NCCL `AllReduce_Sum_bf16_RING_LL`).
With 2 ranks a ring all-reduce is two dependent network transfers (reduce-scatter step, then all-gather step, each
through the sender's NCCL proxy thread, the NIC and the receiver's flag poll; GB10 has no GPUDirect RDMA, so both go
through pinned host FIFOs). An all-gather of the same pair is ONE transfer. `GLM53_DEC_AR1SHOT=1` serves every
all-reduce of a 2-rank group of <= 512 KiB as `pynccl all_gather` + one bf16 add. The sum is **bit-identical** (with
two ranks every element of NCCL's sum is one correctly rounded bf16 addition a + b, which is what torch's bf16 add
computes; tested against the real NCCL all-reduce, adversarial data, eager and inside replayed CUDA graphs). It
removes one network latency L per all-reduce (and up to 2L on the late rank of a skewed pair); it adds one ~1.1 µs add
kernel. **Projection (modelled from the R15 trace, NOT measured: nodeC has no RDMA NIC and one GPU): -0.5 to -1.0 ms
per step, central ≈ -0.7 ms, on structured, prose, coding and ja alike; the review (section 8) widens the low end to
-0.2 ms** (the all-reduce count per step does not depend
on K; the bytes are in NCCL's LL latency regime at every M). nodeC confirms the hop structure: over its loopback
transport the one-shot collective takes exactly half the ring's time at every size (2202 vs 4404 µs).

## 1. Where the all-reduce time goes (R15 rank-0 trace, prose, M = 5)

`ana/exposed.py` (priority timeline: bandwidth-bound kernels > NCCL > small kernels > idle) on the 39 decode steps,
69.0 ms median step:

| exposed time per step | ms |
|---|---|
| bandwidth-bound kernels (EXL3 grouped MoE, FP8 GEMVs, Marlin fc, bf16 GEMMs >= 15 µs) | 54.88 |
| NCCL (102 AR + 2 AG), not hidden by anything | **4.40** |
| small kernels exposed (mhc 1.68, KDA small 1.42, aten glue 1.37, MLA indexer chain 0.99, MoE glue 0.77, small bf16 0.44) | 6.68 |
| GPU idle (host tail, cold NCCL host node: removed in production since by HOSTLOOP + WAKE) | 2.99 |

All-reduce durations (`ana/ar_dist.py`, `ana/ar_excess.py`, 4080 kernels): min 19.2, p05 20.3, p10 20.6, p50 23.8,
p90 54.6 µs; per step 57.5 kernels < 25 µs (both ranks arrive together or rank 0 is the later one) and 44.5 >= 25 µs
(rank 0 waits for rank 1). The excess over a 20 µs floor is 2.2 ms/step, of which 0.93 ms sits at positions 0, 11 and
12 (the drafter-graph and target-graph first collectives and the eager target embedding: host tail and cold host node,
since fixed in production by `GLM53_DEC_HOSTLOOP=1` + `_WAKE=auto`); the rest is GPU-speed skew between the nodes
(01:00 two-rank pair, `docs/logs/hostloop/skew_0100_pair.log`: ~1-10 µs per in-graph collective). Every in-graph NCCL
kernel also starts after a 3.9 µs (p50) GPU gap (0.52 ms/step; every other kernel follows its predecessor after
0.1 µs), see section 5.

## 2. Mechanism (`glm53_ar1shot.py`)

`CudaCommunicator.all_reduce` (the device communicator behind `tensor_model_parallel_all_reduce` and vLLM's
`vllm::all_reduce` custom op, in every graph and eager path of the target, the drafter and the MoE runner) is wrapped:

```
eligible = world_size 2, pynccl enabled, CUDA bf16/fp16/fp32 contiguous, numel*itemsize <= MAX_KB (default 512 KiB)
eligible and agreed -> g = empty([2, n]); pynccl.all_gather(g, x); out = g[0] + g[1]        (one add kernel)
otherwise           -> production's all_reduce, unchanged
```

The decision depends only on the tensor's shape/dtype and the group, identical on both ranks, so both ranks always
issue the same collective. TP-wide agreement: right after `CudaCommunicator.__init__` (collective on every rank, same
order, pynccl's own warm-up all-reduce runs inside it, long before the profile run and any capture) each rank
all-reduces a readiness flag over production's path; the group is served only if every rank is ready. A group built
before the hook (none in production: the plugin registers before `init_device`) agrees lazily at its first eligible
eager call; inside a capture an undecided group takes production's path (counted as `fallback`).

Why the bits are production's: NCCL's 2-rank ring computes each element once, `a + b` in bf16 (`FuncSum` =
`__hadd2`, round-to-nearest-even) on the rank that owns the chunk, and copies it to the peer. torch's bf16 add is
float(a) + float(b) rounded once to bf16; the float sum of two bf16 values is exact or, when the exponents differ by
more than 16, below a quarter ulp of the result, so the single rounding gives the correctly rounded bf16 sum. IEEE
addition is commutative, so g[0] + g[1] on rank 0 and on rank 1 are the same value, equal to NCCL's.

## 3. Exactness evidence (nodeC)

Two vLLM TP ranks = two processes on the one GB10, NCCL's net transport with its proxy thread over loopback sockets
(`NCCL_HOSTID` per rank; P2P/SHM/IB off) — the RoCE pair's code path except the wire.

`tests/decode6/test_ar1shot.py` (`logs/test_ar1shot_r1.log`): **ALL OK**

| test | result |
|---|---|
| A NCCL all_reduce vs one-shot vs bf16(a + b) reference, 88 tensors / 5,677,056 elements: M 1..64 x 4096 bf16 (+ fp16, fp32), random over 2^-40..2^40, exact rounding ties, cancellations, exponent gaps 8..30, subnormals, ±0, overflow | 0 differing (both ranks) |
| B installed wrapper: `tensor_model_parallel_all_reduce` eager M = 1, 5, 8, 64 == production; 2 MiB -> production's ring; CUDA graph with 2 chained captured one-shot calls, 8 replays with new inputs == production | equal (both ranks) |
| C verify mode: 0 differing of 1,277,952 compared elements, production's result served | OK |

`tests/decode6/test_ar1shot_init.py` (production's ORDER: plugin installed before the process group exists, the first
eligible all-reduce of the process inside a capture under vLLM's `GroupCoordinator.graph_capture`;
`logs/test_ar1shot_init_r1.log`): **ALL OK**

| test | result |
|---|---|
| E plugin installed before `init_distributed_environment`; agreement inside `CudaCommunicator.__init__` | `agree_at_init=2` on both ranks (the world group and the TP group, both 2-rank), `agreed(tp)=True`, 0 eager / 0 captured calls before the capture |
| E first eligible all-reduce inside `tp.graph_capture()` + `torch.cuda.graph` | 18 of 18 decode-size calls captured one-shot, `fallback 0`; the 2 MiB call in the same capture took production's ring |
| F 6 replays with new inputs: 8 bf16 row counts M = 1..8 chained twice (the verify block at K 0..7, partial acceptance changes M from step to step) + fp32 + fp16 + the 2 MiB ring call | every output bit-identical to production's all-reduce (mode off, eager), both ranks |

`logs/test_ar1shot_r2.log`: test_ar1shot.py A-C re-run on the final module (install after init = the lazy path).

## 4. Speed

### 4.1 What nodeC can and cannot measure

nodeC has no RDMA device (`/sys/class/infiniband` empty) and one GPU, so the RoCE latency L of a hop cannot be measured
here, and production's GPUs are not to be touched. The loopback-socket rig confirms the structure only: one-shot
2201-2211 µs vs ring 4403-4439 µs per collective at M = 1..64, eager and in graphs (`logs/test_ar1shot_r1.log` D) —
exactly one hop vs two, with the socket transport's polling interval as the "latency".

### 4.2 Model from the production trace

* Ring all-reduce with 2 ranks, arrival of the later rank at s: the later rank finishes at s + 2L, the earlier at
  s + L. One-shot: the later rank finishes at s + ε (the peer's half is already in its FIFO), the earlier at s + L.
  Aligned arrival: 2L + c -> L + c. So each all-reduce saves L, and up to 2L on the later rank (the critical path)
  when the pair is skewed by more than L.
* L from the trace floor: AR p05 20.3 µs = 2L + c with the fixed kernel cost c (LL setup, flag polls, the reduction,
  the store) between 3 and 8 µs -> L ≈ 6.2-8.7 µs. The data time is unchanged (ring: two hops of n/2; one-shot: one
  hop of n).
* The added add kernel: the same `CUDAFunctor_add<bf16>` production runs 62x/step already: 1.0-1.2 µs (+0.1 µs gap).
* Per step: 102 x (-L) + 102 x 1.3 µs = -0.50 … -0.75 ms; the 44.5 skewed all-reduces per step add up to another
  -L each on the critical path (≈ -0.3 ms) -> **-0.5 to -1.0 ms/step, central ≈ -0.7 ms**, the same on structured,
  prose, coding and ja (102 all-reduces per step at every K; 40-64 KiB is far inside NCCL's LL regime).
* Not in the model: whether NCCL picks the same channel count for the all-gather as for the all-reduce at these sizes
  (the R15 trace's eager 1 MiB all-gather ran RING_LL on 8 channels, as the all-reduces), and the extra 40-64 KiB
  allocation per call inside the graph pool (static after capture).

## 5. Measured but not built: the 3.9 µs GPU gap in front of every NCCL kernel

`tests/decode6/mb_nccl_gap.py` (`logs/mb_nccl_gap.log`, `logs/mb_nccl_gap2.log`): two ranks on nodeC (loopback
sockets), a replayed CUDA graph of 20 x [elementwise -> collective -> elementwise], torch.profiler. The gap is
reproduced exactly: **4.0-4.4 µs (p50) before every graph-captured NCCL kernel** (all-reduce and all-gather alike),
0.13 µs between any two other kernels, 0.35-0.51 µs after the NCCL kernel. It does not depend on the transport, so it
is the same cost in production (R15: 3.9 µs p50, 0.52 ms/step). Variants (p50 before the NCCL kernel, rank 0 / 1, AR):

| env | gap | note |
|---|---|---|
| base | 4.40 / 4.00 (2nd run 4.19 / 4.03) | |
| NCCL_MEM_SYNC_DOMAIN=0 | 4.16 / 4.03 | no effect |
| NCCL_LAUNCH_ORDER_IMPLICIT=1 | 8.00 / 7.60 | worse |
| NCCL_MIN/MAX_NCHANNELS=1, =8 | 4.80 / 4.16, 4.22 / 4.61 | no effect |
| NCCL_WORK_ARGS_BYTES=512, =1024 | 3.97 / 5.22, 4.03 / 5.28 | no effect (the 4 KiB kernel-argument block is not it) |
| NCCL_GRAPH_STREAM_ORDERING=0, =1; NCCL_GRAPH_HELPER_DISABLE=1; NCCL_GRAPH_REGISTER=0 | 4.05-5.01 | no effect |
| **NCCL_GRAPH_MIXING_SUPPORT=0** | **2.08 / 3.15** (AG 2.50 / 2.75) | halves it: the cross-stream serialization NCCL captures around every collective |

Not built: `NCCL_GRAPH_MIXING_SUPPORT=0` is only correct when a communicator never has a graph launch outstanding
while a non-captured NCCL call is issued (NCCL docs: "stream ordering is not sufficient"), and production's host loop
issues the eager target-embedding all-reduce and the eager lm_head all-gather while the drafter / target graphs are
still queued. At most ~0.2 ms/step (101 x ~2 µs) would be on the table, against a hang risk. AR1SHOT keeps one
NCCL launch per all-reduce (the all-gather has the same gap), so this gap is neither added nor removed by it.

## 6. Knobs, boot strings, revert

`GLM53_DEC_AR1SHOT` = unset/0 (production) | 1 | verify; `_MAX_KB` (1..16384, default 512); `_LOG` (stats period in
eager all-reduce calls, default 2000, first after 64). `verify` serves production's all-reduce and also runs the
one-shot one, counting differing elements on the GPU (one extra collective per all-reduce: a measurement mode).

Boot, both ranks: `glm53_ar1shot plugin loaded (pid N): GLM53_DEC_AR1SHOT='1' -> installing (mode on)`,
`glm53_ar1shot: hooked CudaCommunicator.all_reduce (mode on, 2-rank groups, <= 512 KiB, dtypes bf16/fp16/fp32)
PROOF mode=on`, `glm53_ar1shot: rank R/2 agreement: every rank ready -> one-shot all-reduce armed (mode on, <= 512
KiB)` (TP group and world group, at communicator construction). Serving (the A/B PROOF): `[glm53-ar1shot] rank R
serving confirmed (mode on): N graph-captured one-shot all-reduces` at the first eager all-reduce after capture; stats
`[glm53-ar1shot] rank R mode on: captured N graph calls, eager E, fallback F`. Never: `glm53_ar1shot: not installed`,
`glm53_ar1shot not loaded`, `agreement: a rank is not ready`, `one-shot all-reduce differs from production's`.
Off: `-> off, production all-reduce unchanged`. Revert: `tools/env_r16.sh off ar1shot` + restart.

## 7. Risks

* A one-rank install would pair an all-gather with an all-reduce (hang). Both ranks run the same image and code and get
  the same env (start.sh forwards the knobs to both, boot_checks compares head and worker); the readiness agreement
  covers a rank that installed but cannot serve, not a rank whose install raised (that rank logs
  `glm53_ar1shot: not installed` — a boot_checks `never` row; restart with the knob off).
* Speed is a model (section 4.2). The operator's A/B decides; `verify` cannot measure speed.

## 8. Adversarial review (2026-10-05, decode6-lat)

Re-run on nodeC on the reviewed module (`docs/logs/ar1shot_review/`): test_ar1shot.py A-D ALL OK (D: one-shot 2201-2221
vs ring 4404-4436 µs on loopback sockets, i.e. 1 hop vs 2 again), test_ar1shot_init.py E/F ALL OK, and the new
`tests/decode6/test_ar1shot_adv.py`:

| test | result |
|---|---|
| G verify mode, first eligible call inside a capture (production's order), 6 replays + 3 eager calls | **failed on the submitted module**: compared=69,632 instead of 712,704 - the counter was created lazily inside the capture, its zeros kernel was recorded into the graph and every replay reset it (the 1-hour verify run would have reported only the elements since the last replay of the first captured graph). **Fixed**: the counter is created eagerly in the agreement (`_agree`), never inside a capture. Re-run: 712,704 / 0 differing on both ranks |
| H production's host-loop mix: 4 target-like graphs (6 layers x [AR -> aux-stream AR (shared expert) -> AR + add], a 3-5 MiB ring AR), 2 drafter-like graphs, replayed without a host sync while an eager one-shot AR (embedding) and an eager 1 MiB all-gather (lm_head) are issued on the same communicator, M changing every iteration, production's NCCL_MIN/MAX_NCHANNELS=8, NCCL_CUMEM_ENABLE=0 | 24/24 iterations bit-identical to production's all-reduce, both ranks, 80 captured one-shot calls, fallback 0. (With all graphs in ONE shared pool replayed out of capture order the drafter outputs differ - identically with production's all-reduce in the graphs, so that is a harness error, kept as the AR1ADV_POOL=1 control) |
| K NaN / inf payloads | NCCL and the one-shot add give the same bit patterns (0x7f80 inf, 0x7fff NaN): verify cannot false-alarm on them |

Other review findings:

* Production uses the image's NCCL 2.30.7 (no LD_PRELOAD in the head container env; `~/nccl-2.30.7` does not exist on
  nodeA), the same library the nodeC tests ran. The R15 trace shows the decode all-reduce as
  `AllReduce_Sum_bf16_RING_LL` grid 5 and the eager 1 MiB all-gather as `AllGather_RING_LL` grid 8 (min 44.9 µs).
* The kit's prev-r16 MANIFEST, mapped as apply_r16.sh maps it, verifies against nodeA's live files (61 overlay/tf files,
  none extra; start.sh 1686a3b2); `~/tf-exl3-deploy16.r16z6rev-kl2` on nodeA passes its MANIFEST. Generator stages
  reproduce c6354b57 / 1686a3b2 (r16z6) and c675a8ec / 24ec973c (r16z6ar). test_boot_checks.sh on the kit: ALL OK
  (231 ok, B.30 11 rows).
* Speed (unchanged: a model). The 20.3 µs floor = 2L + c split is an assumption; if c (kernel start, LL flag polls,
  the reduction) is ~12-14 µs, L is ~3-4 µs and the saving is ~0.2-0.3 ms + the skew term. Expected -0.2 .. -1.0 ms.
* The DEPLOY doc's "temp-0 byte-identical to base, any difference = stop" was not a usable gate: production's temp-0
  output varies run to run (prose 9-10 distinct texts of 10 inside single OFF arms on 2026-10-04). Replaced by the
  nondeterminism-relative rule in DEPLOY_R16Z6AR.md section 2.
* Residual risks: a rank that does not run the module at all (env mismatch, install exception on one rank) pairs the
  other rank's 1-element agreement all-reduce with its own next collective: hang at boot (both ranks get identical env
  and code from start.sh; boot_checks compares them). The eager embedding all-reduce now costs ~2 small allocations +
  one add launch more host time per call (2 calls/step).

