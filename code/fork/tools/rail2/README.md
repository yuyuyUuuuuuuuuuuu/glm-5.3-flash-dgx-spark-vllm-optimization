# Dual PCIe-half RoCE rail for the GLM-5.3 TP=2 link (2026-09-28)

GB10's ConnectX-7 runs in multi-host mode: each QSFP port has one function on PCIe domain 0000 (`enp1s0f*`, `rocep1s0f*`)
and one on domain 0002 (`enP2p1s0f*`, `roceP2p1s0f*`, down by default). Each domain is a PCIe x4 link.
This kit has two cables (nodeA f0 <-> nodeB f0: 10.0.101.x, f1 <-> f1: 10.0.100.x).

ib_write_bw (1 MiB, 4 QPs, 5 s) with GLM stopped, R14:
| link(s) | Gb/s |
|---|---|
| rocep1s0f1 alone (what NCCL used) | 111.9 (PCIe x4 bound) |
| roceP2p1s0f1 alone | 111.9 |
| same cable, both halves (p1s0f1 + P2p1s0f1) | 196.1 (port bound) |
| two cables, both halves (p1s0f1 + P2p1s0f0) | **223.8** |

Production now: HEAD_CX7_IB=WORKER_CX7_IB=rocep1s0f1,roceP2p1s0f0, NCCL_IB_MERGE_NICS=0, GID index 3 on all four.
The P2 functions get 10.0.102.<n>/24 (enP2p1s0f1np1) and 10.0.103.<n>/24 (enP2p1s0f0np0), MTU 9000.

- `glm53-rail2` (installed as /usr/local/sbin/glm53-rail2 with a NOPASSWD rule by `install-glm53-rail2.sh`, run once per
  node with sudo): up | down | persist | status. `persist` wrote /etc/netplan/99-cx7-p2.yaml (applies at boot).
- `r14_rail2.sh`: idle-gated stop, rail up, bandwidth A/B, picks the faster pair, .env switch, start, single-rail fallback.
- `../prodcheck/fix_gids.sh`: HEAD_CX7_IB / WORKER_CX7_IB may be comma lists; every device must resolve to one index.
- Rollback: HEAD_CX7_IB=WORKER_CX7_IB=rocep1s0f1 and NCCL_IB_MERGE_NICS=0 in .env, restart.
Result (R13 -> R14, which also dropped the NCCL tuner): real-text prefill 8.5k 1299 -> 1370, 15.3k 1324 -> 1412 tok/s,
decode unchanged (73.8 / 83.4 ms/step), long-context KL at noise.
