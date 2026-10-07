#!/usr/bin/env python3
"""deploy-r16 launcher start.sh = production start.sh (sha256 ed7dcd8e...) + the APC short-suffix diff
(git -C <launcher-apc> diff 09c9180 main -- start.sh) + env passthrough of every R16 bundle knob to BOTH ranks:
  - head:   `-e NAME="${NAME:-}" \\` lines right after the head docker run's GLM53_KPOOL_SEED_STRIDE line
  - worker: the names appended to the serve_env_names `for v in ... GLM53_KPOOL_SEED_STRIDE; do` loop
Every anchor must occur exactly once or nothing is written; the result must pass `bash -n`, equal
<launcher-apc>/start.sh (main) outside the passthrough edit, and grep must find every knob in both places (exactly one
head line, exactly one worker-list occurrence).
r16j: that R16 stage must be the r16i kit's start.sh byte for byte (sha256 b721de12..., = blockverify's KIT_SHA), and
the block-verification switch GLM53_REJECTION_METHOD is then added by blockverify's own edits E1-E5 (imported from
tools/blockverify/make_start_sh.py, not copied): the spec JSON of both inner scripts, validate_numeric_config, the head
`-e` line right after GLM53_SPEC_RESAMPLE_INDEPENDENT, the worker serve_env_names entry, the note. Same contract as
blockverify's generator (anchor counts, bash -n, knob placement head 1 / worker 1 / spec 2, only the edits changed), and
the result must equal the committed launcher/start.sh (cea89226...) when that file is present.
The switch is NOT an R16 knob (not in KNOBS_DEFAULT: it sits next to GLM53_SPEC_RESAMPLE_INDEPENDENT, not after
GLM53_KPOOL_SEED_STRIDE) and not an env.r16 value line (tools/deploy16/env.r16 `#switch`): unset = standard.
r16k: the r16j stage (sha256 cea89226..., pinned: the start.sh production runs since 2026-09-30) + the kdaqkv knob
GLM53_KDA_STRIDED_QKV (docs/KDA_STRIDED_QKV.md) forwarded to BOTH ranks: K1 validate_numeric_config (empty/unset = stock;
otherwise exactly 0 or 1; 1 also needs overlay/tf/overlay/patch_kda_strided_qkv.py and a patch_tf_bundle.py that runs it),
K2 head `-e` right after GLM53_KPOOL_RING, K3 the worker serve_env_names loop right after GLM53_KPOOL_RING, K4 a note.
Same contract (anchor counts, bash -n, knob placement head 1 / worker 1 / validation 1, only the edits changed), and the
result must equal the committed launcher/start.sh when that file is present. Without the knob in the environment the
containers get GLM53_KDA_STRIDED_QKV="" and the bundle skips the overlay: stock.
r16l: the r16k stage + the kpooldown knob GLM53_KPOOL_DROP_LOWEST (docs/KPOOL_DROP_LOWEST.md) = the committed
launcher/start.sh of r16l (L1-L4, same contract).
r16m: the r16l stage (971c984c..., pinned) + the flashkda knob GLM53_KDA_FLASHKDA (docs/KDA_FLASHKDA.md) forwarded to
BOTH ranks: F1 validation (empty/unset = stock; exactly 0 or 1; 1 also needs overlay/tf/overlay/patch_flashkda.py and a
patch_tf_bundle.py that runs it), F2 head `-e` right after GLM53_KPOOL_DROP_LOWEST, F3 the worker serve_env_names loop
right after GLM53_KPOOL_DROP_LOWEST, F4 a note. Same contract; the result must equal the committed launcher/start.sh.
r16msp: the r16l stage + the mHC sequence-parallel knob GLM53_MHC_SP (docs/MHC_SP.md), M1-M4, same contract (the
standalone mhcsp stage; r16x builds the same M texts on the r16n stage's F results instead).
r16e4: the r16l stage + the e4m3 routed-MoE knob GLM53_MOE_E4M3 (docs/MOE_E4M3.md), E1-E4, same contract (the
standalone moee4m3 stage; r16x builds the same E texts on the r16x stage's M results instead).
r16x: the r16n stage + mhcsp + moee4m3 (the combined kit: the three reviewed features behind their own switches, each
default OFF). The chain is fixed: flashkda's F1-F4 (== the r16n stage), then the M edits anchored on the F results,
then the E edits anchored on the M results - so a regenerate reproduces the committed launcher/start.sh byte for byte.
r16y: the r16x stage + the w8a8 switch GLM53_DENSE_W8A8 (docs/DENSE_W8A8.md), YA1-YA4 anchored on the E results
                (the same chain continued: F -> M -> E -> YA) - the four feature switches of the combined kit, each
                default OFF.
r16z2: the r16z stage + TWO edits, in the chain's order (F -> M -> E -> YA -> ZV -> ZS -> ZW -> ZD):
                ZW1-ZW4 the w8a82 knobs (docs/DENSE_W8A8_2.md): GLM53_DENSE_W8A8_GEMM (unset/custom = the custom
                CUTLASS SM120 GEMM; cutlass_mm = the image's cutlass_scaled_mm in pieces) and GLM53_DENSE_W8A8_ONLY
                (unset = every served projection; else a comma list of the module's PROJ_NAMES) - only read when
                GLM53_DENSE_W8A8=1, and ZD1-ZD4 the e4m3 down-projection width GLM53_MOE_E4M3_DOWN (unset/empty/e4m3
                = the original path; f16 = the fused variant 16) - only read when GLM53_MOE_E4M3=1; each forwarded
                to BOTH ranks, default = production byte for byte.
r16z: the r16y stage + TWO edits, in the chain's order (F -> M -> E -> YA -> ZV -> ZS):
                ZV1-ZV4 the FlashKDA BUILD switch GLM53_KDA_FLASHKDA_V (docs/KDA_FLASHKDA3.md): unset/1 = the
                shipped r16x build (production bytes today), 2 = the fkda2 precision build, 3 = the fkda3 build
                (all three stage under the same site-packages names; only read when GLM53_KDA_FLASHKDA=1), and
                ZS1-ZS4 the pipelined-SP switch GLM53_MHC_SP2 (docs/MHC_SP2.md; requires GLM53_MHC_SP=1) - each
                forwarded to BOTH ranks, default = production byte for byte.
r16z3: the r16z2 stage + TWO edits, in the chain's order (F -> M -> E -> YA -> ZV -> ZS -> ZW -> ZD -> ZF -> ZL):
                ZF1-ZF4 the fused16 switch GLM53_MOE_FUSED16 (docs/MOE3.md): the P16 routed-MoE prefill - production
                arithmetic on faster schedules (h2 bit-identical); 1 = the bundle overlay patch_moe_fused16.py (copies
                glm53_moe_fused16.py + the SHARED e4m3 extension into site-packages and arms integrate.py, its block
                goes BEFORE glm53_moe_e4m3's) - and ZL1-ZL4 the layer lists GLM53_MOE_E4M3_LAYERS (only those model
                layers run e4m3; unset = every MoE layer) and GLM53_MOE_E4M3_DOWN_LAYERS (exactly those layers the
                f16 down; unset = GLM53_MOE_E4M3_DOWN decides) - comma lists / ranges of the model's MoE layer
                indices 3..44, only read when GLM53_MOE_E4M3=1; each knob forwarded to BOTH ranks, default =
                production byte for byte.
r16z4: the r16z3 stage + FOUR edits, in the chain's order (F -> M -> E -> YA -> ZV -> ZS -> ZW -> ZD -> ZF -> ZL
                -> ZT -> ZP -> ZG -> ZH):
                ZT1-ZT4 the opt-moe dials of the e4m3 path (docs/OPT_MOE.md), only read when GLM53_MOE_E4M3=1:
                GLM53_MOE_E4M3_ACC (unset/empty/f32 = the fp32 accumulator, production's arithmetic; bf16 = the
                bf16 accumulator), GLM53_MOE_E4M3_FOLD_SHARED (0|1; 1 REQUIRES GLM53_MOE_E4M3_ACC=bf16 and is
                refused here otherwise - the module would refuse the whole e4m3 install) and
                GLM53_MOE_E4M3_TOKGATHER (0|1; unset/1 = on = the designed default: one gathered gate/up row per
                token, bitwise the per-pair result; 0 = the per-pair gather) - ACC/FOLD default OFF (the MoE
                refutation is still running; a rejection arrives as a follow-up), TOKGATHER default on as designed;
                ZP1-ZP4 the MLA prefill fused index pass GLM53_MLA_PREFILL_FUSED_INDEX (docs/OPT_DENSE.md; 0|1,
                unset/1 = on - the deliberate default change; 0 = production's triton_convert + clamp + copy
                chain, the revert lever; read by the site glm53_mla_prefill.py this kit ships);
                ZG1-ZG4 the fp8 sequence-parallel all-gather GLM53_DENSE_W8A8_FP8AG (docs/OPT_DENSE.md; 0|1, only
                read when GLM53_DENSE_W8A8=1; a PAIRED collective - boot_checks pair-gates "FP8 all-gather
                installed" on BOTH ranks before traffic); and
                ZH1-ZH4 the hi+lo W8A8 knobs GLM53_DENSE_W8A8_HILO (a comma list of "<group>.<proj>:<channels>",
                channels a multiple of 16 in 16..2048, draft.fc not accepted) and GLM53_DENSE_W8A8_HILO_SEL
                (call|first; default call), only read when GLM53_DENSE_W8A8=1 -
                each forwarded to BOTH ranks, default = production byte for byte.
r16z6: the r16z5 stage + TWO edits, in the chain's order (... -> ZN -> KL -> VT):
                KL1-KL4 the decode KDA_LAZY knobs (docs/DEC_KDA_LAZY.md): GLM53_DEC_KDA_LAZY (0|1; 1 = the one
                KDA recurrent-state store per verify step, site glm53_kda_lazy.py) + its self-check dials
                GLM53_DEC_KDA_LAZY_VERIFY (commits checked, default 64) / _VERIFY_EVERY (default 1024; both
                unset/empty or a non-negative integer; the module ships in the bundle site/, not an overlay), and
                VT1-VT4 the verify-trimming collector GLM53_DEC_VTRIM_STATS (verify calls between .npy flushes;
                unset/empty/0 = off, log-only) + GLM53_DEC_VTRIM_STATS_CAP / _FILE (a positive integer / a path;
                site glm53_vtrim_stats.py) - each forwarded to BOTH ranks, default = production byte for byte.
r16z6sv: the r16z6 stage + SV1-SV4 (GLM53_SPEC_VTRIM + _TAU / _MIN / _LOG, docs/SPEC_VTRIM.md).
r16z7: the r16z6sv stage + PT1-PT4 (docs/PREFIX_HIT_TAIL.md, docs/DEPLOY_R16Z7.md): GLM53_KPOOL_TAIL_POSITIONS
                (unset/empty/0 = stock; 2 = per-request kpool tail rings written INTO the persistent slot buffer,
                graph-safe; 1 is REFUSED here: FULL CUDA graph decode replays never read it) and
                GLM53_MAMBA_ALIGN_SEED (0|1; 1 = a prefix hit seeds its KDA column in mamba blocks; production-
                neutral hardening) - overlays run by patch_tf_bundle.py, forwarded to BOTH ranks (a one-rank tail
                fix silently degrades quality), default = production byte for byte.
Usage: make_start_sh.py <production start.sh> <launcher-apc repo> <out start.sh> [--knobs "A B C"] [--stage r16|r16j|r16k|r16l|r16m|r16x|r16y|r16z|r16z2|r16z3|r16z4|r16z6|r16z6sv|r16z7|r16z8|r16z8p]
  --stage r16   write the R16 stage only (production + APC diff + passthrough; with the default knobs = b721de12)
  --stage r16j  the R16 stage + the blockverify switch (= cea89226, production's start.sh)
  --stage r16k  the r16j stage + the kdaqkv knob (= the committed launcher/start.sh of r16k)
  --stage r16l  the r16k stage + the kpooldown switch (= the committed launcher/start.sh of r16l)
  --stage r16m  the r16l stage + the flashkda switch (the r16m kit's start.sh)
  --stage r16n  byte-identical to r16m: the flashkda stage itself is unchanged by the r16n review fixes
                (B1 kda_conv allowlist / D1 boot-time workspace reservation are overlay + tools changes)
  --stage r16x  (default) the r16n stage + the mhcsp switch GLM53_MHC_SP + the moee4m3 switch GLM53_MOE_E4M3
                (= the committed launcher/start.sh of the combined kit)
Every stage fails closed on any anchor drift."""
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PROD_SHA = "ed7dcd8e13ec523992974c894f010e30889b252dfa88fd34868fc104fbf29b84"
KNOBS_DEFAULT = """GLM53_PREFILL_QUICKWINS GLM53_PREFILL_QUICKWINS_MIN_T
GLM53_MLA_PREFILL GLM53_MLA_PREFILL_MIN_TOKENS GLM53_MLA_PREFILL_MIXED GLM53_MLA_PREFILL_VARIANT
GLM53_DEC_FP8ROOF GLM53_DEC_FP8ROOF_TABLE GLM53_DEC_FP8ROOF_PF GLM53_DEC_FP8ROOF_PF_MIB GLM53_DEC_FP8ROOF_PF_CTAS
GLM53_DEC_FP8ROOF_PF_POL GLM53_DEC_FP8ROOF_PF_MAX_M
GLM53_DEC_MOEGLUE GLM53_DEC_MOEGLUE_PREFETCH GLM53_DEC_MOEGLUE_WARM GLM53_DEC_MOEGLUE_WARM_SET GLM53_DEC_MOEGLUE_WARM_MIB
GLM53_DEC_MOEGLUE_WARM_BLOCKS GLM53_DEC_MOEGLUE_WARM_MAX_M
GLM53_DEC_HOSTLOOP GLM53_DEC_HOSTLOOP_VERIFY GLM53_DEC_HOSTLOOP_VERIFY_EVERY GLM53_DEC_HOSTLOOP_METER
GLM53_DEC_HOSTLOOP_WAKE GLM53_DEC_HOSTLOOP_WAKE_TICK_US GLM53_DEC_HOSTLOOP_WAKE_PIN GLM53_DEC_PROF_DIAG
GLM53_DEC_SMALLOPS GLM53_DEC_SMALLOPS_KINDS
GLM53_KPOOL_RING"""

WORKER_OLD = ("             NCCL_TUNER_PLUGIN NCCL_TUNER_CONFIG_FILE GLM53_PREFILL_FUSED_CAP GLM53_FP8_LARGE_M "
              "GLM53_KPOOL_SEED_STRIDE; do  # [tf-exl3-fork]\n")
HEAD_ANCHOR = '        -e GLM53_KPOOL_SEED_STRIDE="${GLM53_KPOOL_SEED_STRIDE:-}" \\\n'
NOTE_ANCHOR = "# [tf-exl3-fork] head env passthrough added after GLM53_DENSE_FP8 in the head docker run.\n"
NOTE = ("# [tf-exl3-fork r16] every R16 bundle knob (quickwins, MLA prefill, DEC_FP8ROOF, DEC_MOEGLUE, DEC_HOSTLOOP,\n"
        "#   DEC_SMALLOPS, KPOOL_RING) is forwarded to BOTH ranks: head `-e` lines after GLM53_KPOOL_SEED_STRIDE, worker\n"
        "#   serve_env_names loop. Unset knobs arrive as \"\" = the module default (deploy-r16 tests/r16/test_r16_plugins.py P4).\n")


def main() -> int:
    argv = sys.argv[1:]
    knobs = KNOBS_DEFAULT.split()
    stage = "r16x"
    for opt in ("--knobs", "--stage"):
        if opt in argv:
            i = argv.index(opt)
            has_val = i + 1 < len(argv) and not argv[i + 1].startswith("--")
            val = argv[i + 1] if has_val else ""
            del argv[i:i + (2 if has_val else 1)]
            if opt == "--knobs":
                knobs = val.split()
            elif opt == "--stage":
                stage = val
    if stage not in ("r16", "r16j", "r16k", "r16l", "r16m", "r16n", "r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z6dl", "r16z8", "r16z8p"):
        sys.exit(f"ABORT: --stage must be r16, r16j, r16k, r16l, r16m, r16n, r16x, r16y, r16z, r16z2, r16z3, r16z4, r16z5, r16z6, r16z6sv, r16z7, r16z6ar, r16z7ar, r16z6dl, r16z8 or r16z8p (got {stage})")
    prod, apc, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    s0 = prod.read_bytes()
    if hashlib.sha256(s0).hexdigest() != PROD_SHA:
        sys.exit(f"ABORT: {prod} is not production start.sh {PROD_SHA[:16]}")
    diff = subprocess.run(["git", "-C", str(apc), "diff", "09c9180", "main", "--", "start.sh"], check=True,
                          capture_output=True).stdout
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "start.sh").write_bytes(s0)
        subprocess.run(["git", "init", "-q", td], check=True)
        p = subprocess.run(["git", "-C", td, "apply", "--whitespace=nowarn", "-"], input=diff, capture_output=True)
        if p.returncode:
            sys.exit(f"ABORT: APC diff does not apply: {p.stderr.decode()[:500]}")
        s1 = (Path(td) / "start.sh").read_text()
    main_sh = subprocess.run(["git", "-C", str(apc), "show", "main:start.sh"], check=True, capture_output=True).stdout
    if s1.encode() != main_sh:
        sys.exit("ABORT: production start.sh + APC diff != launcher-apc main start.sh")
    for name, a in (("worker list", WORKER_OLD), ("head -e", HEAD_ANCHOR), ("note", NOTE_ANCHOR)):
        n = s1.count(a)
        if n != 1:
            sys.exit(f"ABORT: {name} anchor found {n}x (need 1)")
    for k in knobs:
        if re.search(rf"\b{k}\b", s1):
            sys.exit(f"ABORT: {k} already in start.sh")
    lines, cur = [], "            "
    for k in knobs:
        if len(cur) + len(k) + 3 > 116:
            lines.append(cur + " \\\n")
            cur = "            "
        cur += " " + k
    worker_new = WORKER_OLD.replace("; do  # [tf-exl3-fork]\n", " \\\n") + "".join(lines) + cur + \
        "; do  # [tf-exl3-fork]\n"
    head_new = HEAD_ANCHOR + "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in knobs)
    s2 = s1.replace(WORKER_OLD, worker_new, 1).replace(HEAD_ANCHOR, head_new, 1).replace(
        NOTE_ANCHOR, NOTE_ANCHOR + NOTE, 1)
    # --- verify
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n: {r.stderr.decode()}")
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    if not m:
        sys.exit("ABORT: serve_env_names loop not found after the edit")
    wl = m.group(1).replace("\\\n", " ").split()
    bad = []
    for k in knobs:
        nh = len(re.findall(rf'^        -e {k}="\$\{{{k}:-\}}" \\$', s2, re.M))
        nw = wl.count(k)
        if nh != 1 or nw != 1:
            bad.append(f"{k}: head {nh} worker {nw}")
    if bad:
        sys.exit("ABORT: passthrough check: " + "; ".join(bad))
    msg = (f"APC diff applied (== launcher-apc main {hashlib.sha256(main_sh).hexdigest()[:16]}); {len(knobs)} knobs "
           f"forwarded to the head docker run and the worker serve_env_names loop (each exactly once in each); bash -n ok")
    if stage in ("r16j", "r16k", "r16l", "r16m", "r16n", "r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, bmsg = add_blockverify(s2)
        msg += "; " + bmsg
    if stage in ("r16k", "r16l", "r16m", "r16n", "r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, kmsg = add_kdaqkv(s2)
        msg += "; " + kmsg
    if stage in ("r16l", "r16m", "r16n", "r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, pmsg = add_kpooldown(s2, expect_committed=stage == "r16l")
        msg += "; " + pmsg
    if stage in ("r16m", "r16n", "r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, fmsg = add_flashkda(s2)
        msg += "; " + fmsg
    if stage in ("r16x", "r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, spmsg = add_mhcsp(s2)
        msg += "; " + spmsg
        s2, emsg = add_moee4m3(s2)
        msg += "; " + emsg
    if stage in ("r16y", "r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, wmsg = add_dense_w8a8(s2)
        msg += "; " + wmsg
    if stage in ("r16z", "r16z2", "r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, zmsg = add_fkda_version(s2)
        msg += "; " + zmsg
        s2, zsmsg = add_mhcsp2(s2)
        msg += "; " + zsmsg
    if stage == "r16z2":
        s2, zwmsg = add_w8a82(s2)
        msg += "; " + zwmsg
        s2, zdmsg = add_moe_down(s2)
        msg += "; " + zdmsg
        msg += "; " + wmsg
    if stage in ("r16z3", "r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, zwmsg = add_w8a82(s2)
        msg += "; " + zwmsg
        s2, zdmsg = add_moe_down(s2)
        msg += "; " + zdmsg
        s2, zfmsg = add_moe_fused16(s2)
        msg += "; " + zfmsg
        s2, zlmsg = add_moe_layers(s2)
        msg += "; " + zlmsg
        msg += "; " + wmsg
    if stage in ("r16z4", "r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, ztmsg = add_moe_opt(s2)
        msg += "; " + ztmsg
        s2, zpmsg = add_mla_fused_index(s2)
        msg += "; " + zpmsg
        s2, zgmsg = add_w8a8_fp8ag(s2)
        msg += "; " + zgmsg
        s2, zhmsg = add_w8a8_hilo(s2)
        msg += "; " + zhmsg
    if stage in ("r16z5", "r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, xlmsg = add_exactlens(s2)
        msg += "; " + xlmsg
        s2, zumsg = add_kdamhc_kvrows(s2)
        msg += "; " + zumsg
        s2, zmmsg = add_mhc_fused(s2)
        msg += "; " + zmmsg
        s2, zkmsg = add_w8a8_skip(s2)
        msg += "; " + zkmsg
        s2, zwmsg = add_moe_mainloop(s2)
        msg += "; " + zwmsg
    if stage in ("r16z6", "r16z6dl", "r16z6sv", "r16z7", "r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, klmsg = add_kda_lazy(s2)
        msg += "; " + klmsg
        s2, vtmsg = add_vtrim(s2)
        msg += "; " + vtmsg
    if stage in ("r16z6sv", "r16z7", "r16z7ar", "r16z8", "r16z8p"):
        s2, svmsg = add_specvtrim(s2)
        msg += "; " + svmsg
    if stage in ("r16z7", "r16z7ar", "r16z8", "r16z8p"):
        s2, ptmsg = add_kpool_tail(s2)
        msg += "; " + ptmsg
    if stage in ("r16z6ar", "r16z7ar", "r16z8", "r16z8p"):
        s2, armsg = add_ar1shot(s2)
        msg += "; " + armsg
    if stage in ("r16z6dl", "r16z8", "r16z8p"):   # r16z6dl = r16z6 + DL1-DL4 (decode5); r16z8 = r16z7ar + DL1-DL4
        s2, dlmsg = add_dlmh(s2)
        msg += "; " + dlmsg
    if stage == "r16z8p":   # r16z8p = r16z8 + PP1-PP4 (GLM53_MLA_PLAN_PIN) + AB1-AB2 (ABLIT honoured from .env)
        s2, ppmsg = add_planpin(s2)
        msg += "; " + ppmsg
        s2, abmsg = add_ablit_env(s2)
        msg += "; " + abmsg
    out.write_text(s2)
    out.chmod(0o755)
    print(f"{out}: sha256 {hashlib.sha256(s2.encode()).hexdigest()}; stage {stage}; {msg}")
    return 0


def load_blockverify():
    """tools/blockverify/make_start_sh.py as a module (its main() is not run: no file is written by the import)."""
    path = REPO / "tools" / "blockverify" / "make_start_sh.py"
    spec = importlib.util.spec_from_file_location("blockverify_make_start_sh", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_blockverify(s: str) -> tuple[str, str]:
    """R16 stage (must be the r16i kit start.sh, b721de12) -> + blockverify's E1-E5, with blockverify's checks."""
    bv = load_blockverify()
    sha = hashlib.sha256(s.encode()).hexdigest()
    if sha != bv.KIT_SHA:
        sys.exit(f"ABORT: the R16 stage is {sha[:16]}, not the r16i kit start.sh {bv.KIT_SHA[:16]} that blockverify's "
                 f"edits were made and reviewed against")
    if re.search(rf"\b{bv.KNOB}\b", s):
        sys.exit(f"ABORT: {bv.KNOB} already in the R16 stage")
    for name, old, _new, n in bv.EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: blockverify {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in bv.EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the blockverify edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {bv.KNOB}="${{{bv.KNOB}:-}}" \\\n')
    worker = s2[s2.index("local -a serve_env_names=()"):s2.index('serve_env+=" -e $v=')].count(bv.KNOB)
    spec = s2.count(f'os.environ.get("{bv.KNOB}") or "standard"')
    if (head, worker, spec) != (1, 1, 2):
        sys.exit(f"ABORT: {bv.KNOB} placement head={head} worker={worker} spec={spec} (need 1, 1, 2)")
    back = s2
    for _name, old, new, n in bv.EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the blockverify stage differs from the R16 stage outside E1-E5")
    got = hashlib.sha256(s2.encode()).hexdigest()
    if got != R16J_SHA:
        sys.exit(f"ABORT: the r16j stage is {got[:16]}, not blockverify's reviewed start.sh {R16J_SHA[:16]} "
                 f"(the r16j kit's, production's since 2026-09-30)")
    return s2, (f"R16 stage == r16i kit start.sh {bv.KIT_SHA[:16]}; + blockverify E1-E5 ({bv.KNOB}: head -e x{head}, "
                f"worker list x{worker}, spec JSON x{spec}) == r16j kit start.sh {R16J_SHA[:16]}")


# ---- r16k: the kdaqkv knob (docs/KDA_STRIDED_QKV.md) on top of the r16j start.sh
R16J_SHA = "cea8922692a9af01c4f76a58189ae9bae662fcc48ee792006996158288ae5481"
R16K_SHA = "7736abf8797a471a709bebb8cf0d29b3047d1924af612c49752f10274d9b039a"
KDA_KNOB = "GLM53_KDA_STRIDED_QKV"
K1_OLD = '    _glm53_validate_bool_flag GLM53_KDA_BF16_LARGE_M "${GLM53_KDA_BF16_LARGE_M-0}" || return\n'
K1_NEW = K1_OLD + """    # [kdaqkv] vLLM #55736 KDA strided decode inputs (docs/KDA_STRIDED_QKV.md), on both ranks. Unset/empty/0 = stock
    # (the bundle leaves the two FLA files untouched); 1 = the overlay patch_kda_strided_qkv.py, run by patch_tf_bundle.py.
    # Anything else would stop both containers inside the bundle, so it is refused here with its name.
    if [ -n "${GLM53_KDA_STRIDED_QKV:-}" ]; then
        _glm53_validate_bool_flag GLM53_KDA_STRIDED_QKV "$GLM53_KDA_STRIDED_QKV" || return
    fi
    if [ "${GLM53_KDA_STRIDED_QKV:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_kda_strided_qkv.py" ]; then
            echo "GLM53_KDA_STRIDED_QKV=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_kda_strided_qkv.py" >&2
            return 2
        fi
        if ! grep -qF '("patch_kda_strided_qkv.py", "GLM53_KDA_STRIDED_QKV")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_KDA_STRIDED_QKV=1 requires $TF_BUNDLE_PATCH_HOST to run patch_kda_strided_qkv.py (the kdaqkv patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
K2_OLD = '        -e GLM53_KPOOL_RING="${GLM53_KPOOL_RING:-}" \\\n'
K2_NEW = K2_OLD + '        -e GLM53_KDA_STRIDED_QKV="${GLM53_KDA_STRIDED_QKV:-}" \\\n'
K3_OLD = "             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING; do  # [tf-exl3-fork]\n"
K3_NEW = "             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV; do  # [tf-exl3-fork]\n"
K4_OLD = "# [tf-exl3-fork] bundle: TensorFold EXL3 MoE kernels + resample-noise fix + drafter FP8 (inert unless its knobs are set)\n"
K4_NEW = ("# [kdaqkv] GLM53_KDA_STRIDED_QKV=1 (unset/empty/0 = stock) -> the bundle overlay patch_kda_strided_qkv.py (vLLM\n"
          "#   #55736: the KDA recurrent decode reads q/k/v/beta token-strided, no .contiguous() copies) on BOTH ranks:\n"
          "#   head -e after GLM53_KPOOL_RING, worker serve_env_names. docs/KDA_STRIDED_QKV.md of tf-exl3-fork.\n") + K4_OLD
K_EDITS = (("K1 validation", K1_OLD, K1_NEW, 1), ("K2 head -e", K2_OLD, K2_NEW, 1), ("K3 worker list", K3_OLD, K3_NEW, 1),
           ("K4 note", K4_OLD, K4_NEW, 1))


def add_kdaqkv(s: str) -> tuple[str, str]:
    """r16j stage (cea89226) -> + K1-K4, with blockverify's kind of checks; the result is the pinned r16k start.sh."""
    sha = hashlib.sha256(s.encode()).hexdigest()
    if sha != R16J_SHA:
        sys.exit(f"ABORT: the r16j stage is {sha[:16]}, not {R16J_SHA[:16]} that the kdaqkv edits were made against")
    if re.search(rf"\b{KDA_KNOB}\b", s):
        sys.exit(f"ABORT: {KDA_KNOB} already in the r16j stage")
    for name, old, _new, n in K_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kdaqkv {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in K_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kdaqkv edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {KDA_KNOB}="${{{KDA_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(KDA_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {KDA_KNOB} "${KDA_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {KDA_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in K_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kdaqkv stage differs from the r16j stage outside K1-K4")
    sha2 = hashlib.sha256(s2.encode()).hexdigest()
    if sha2 != R16K_SHA:
        sys.exit(f"ABORT: the kdaqkv stage is {sha2[:16]}, not the reviewed r16k start.sh {R16K_SHA[:16]}")
    return s2, (f"+ kdaqkv K1-K4 ({KDA_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}) "
                f"== r16k kit start.sh {R16K_SHA[:16]}")


# ---- r16l (poolfix): the kpooldown knob (docs/KPOOL_DROP_LOWEST.md) on top of the r16k start.sh
PD_KNOB = "GLM53_KPOOL_DROP_LOWEST"
L1_OLD = K1_NEW
L1_NEW = L1_OLD + """    # [kpooldown] the kpool indexer drops the LOWEST-scored pool of the top-k (docs/KPOOL_DROP_LOWEST.md), on both
    # ranks. Unset/empty/0 = stock (production's last-column truncation drops an ARBITRARY pool per run: the top-k
    # ops return a deterministic SET in a nondeterministic ORDER); 1 = the overlay patch_kpool_drop_lowest.py, run
    # by patch_tf_bundle.py. Anything else would stop both containers inside the bundle, so it is refused here.
    if [ -n "${GLM53_KPOOL_DROP_LOWEST:-}" ]; then
        _glm53_validate_bool_flag GLM53_KPOOL_DROP_LOWEST "$GLM53_KPOOL_DROP_LOWEST" || return
    fi
    if [ "${GLM53_KPOOL_DROP_LOWEST:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_kpool_drop_lowest.py" ]; then
            echo "GLM53_KPOOL_DROP_LOWEST=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_kpool_drop_lowest.py" >&2
            return 2
        fi
        if ! grep -qF '("patch_kpool_drop_lowest.py", "GLM53_KPOOL_DROP_LOWEST")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_KPOOL_DROP_LOWEST=1 requires $TF_BUNDLE_PATCH_HOST to run patch_kpool_drop_lowest.py (the r16l patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
L2_OLD = K2_NEW
L2_NEW = L2_OLD + '        -e GLM53_KPOOL_DROP_LOWEST="${GLM53_KPOOL_DROP_LOWEST:-}" \\\n'
L3_OLD = K3_NEW
L3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV "
          "GLM53_KPOOL_DROP_LOWEST; do  # [tf-exl3-fork]\n")
L4_OLD = K4_NEW
L4_NEW = ("# [kpooldown] GLM53_KPOOL_DROP_LOWEST=1 (unset/empty/0 = stock) -> the bundle overlay patch_kpool_drop_lowest.py:\n"
          "#   the kpool indexer keeps the select_k-1 HIGHEST-scored pools before expand_pools_and_append_tail, dropping\n"
          "#   the lowest, deterministically (no host sync, FULL-graph safe), instead of production's arbitrary one. On\n"
          "#   BOTH ranks: head -e after GLM53_KDA_STRIDED_QKV, worker serve_env_names. docs/KPOOL_DROP_LOWEST.md.\n") + K4_OLD
L_EDITS = (("L1 validation", L1_OLD, L1_NEW, 1), ("L2 head -e", L2_OLD, L2_NEW, 1), ("L3 worker list", L3_OLD, L3_NEW, 1),
           ("L4 note", L4_OLD, L4_NEW, 1))


def add_kpooldown(s: str, expect_committed: bool = True) -> tuple[str, str]:
    """r16k stage (pinned R16K_SHA) -> + L1-L4, with kdaqkv's kind of checks;
    == committed launcher/start.sh while that file IS the r16l stage (the
    r16m stage builds on it, so expect_committed=False there and add_flashkda
    does the committed-file check on its own result)."""
    sha = hashlib.sha256(s.encode()).hexdigest()
    if sha != R16K_SHA:
        sys.exit(f"ABORT: the r16k stage is {sha[:16]}, not {R16K_SHA[:16]} that the kpooldown edits were made against")
    if re.search(rf"\b{PD_KNOB}\b", s):
        sys.exit(f"ABORT: {PD_KNOB} already in the r16k stage")
    for name, old, _new, n in L_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kpooldown {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in L_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kpooldown edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {PD_KNOB}="${{{PD_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(PD_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {PD_KNOB} "${PD_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {PD_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in L_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kpooldown stage differs from the r16k stage outside L1-L4")
    got = hashlib.sha256(s2.encode()).hexdigest()
    if got != R16L_SHA:
        sys.exit(f"ABORT: the kpooldown stage is {got[:16]}, not the reviewed r16l kit start.sh {R16L_SHA[:16]}")
    return s2, (f"+ kpooldown L1-L4 ({PD_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}) "
                f"== r16l kit start.sh {R16L_SHA[:16]}")



# ---- r16m: the flashkda knob (docs/KDA_FLASHKDA.md) on top of the r16l start.sh
R16L_SHA = "971c984c0c8ba470e73bb71b27f990e99dbc95b384b9c6117a21799475be72ac"
FK_KNOB = "GLM53_KDA_FLASHKDA"
F1_OLD = L1_NEW
F1_NEW = F1_OLD + """    # [flashkda] FlashKDA 17a037d (fp32 recurrent state, vllm #58846) for the KDA chunked prefill
    # (docs/KDA_FLASHKDA.md), on both ranks. Unset/empty/0 = stock (the Triton chunk_kda_with_fused_gate
    # chain, byte for byte upstream; the _flashkda_fp32_C extension is not installed either); 1 = the
    # overlay patch_flashkda.py, run by patch_tf_bundle.py. Anything else would stop both containers
    # inside the bundle, so it is refused here with its name.
    if [ -n "${GLM53_KDA_FLASHKDA:-}" ]; then
        _glm53_validate_bool_flag GLM53_KDA_FLASHKDA "$GLM53_KDA_FLASHKDA" || return
    fi
    if [ "${GLM53_KDA_FLASHKDA:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_flashkda.py" ]; then
            echo "GLM53_KDA_FLASHKDA=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_flashkda.py" >&2
            return 2
        fi
        if ! grep -qF '("patch_flashkda.py", "GLM53_KDA_FLASHKDA")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_KDA_FLASHKDA=1 requires $TF_BUNDLE_PATCH_HOST to run patch_flashkda.py (the r16m patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
F2_OLD = L2_NEW
F2_NEW = F2_OLD + '        -e GLM53_KDA_FLASHKDA="${GLM53_KDA_FLASHKDA:-}" \\\n'
F3_OLD = L3_NEW
F3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
          "GLM53_KDA_FLASHKDA; do  # [tf-exl3-fork]\n")
F4_OLD = K4_OLD
F4_NEW = ("# [flashkda] GLM53_KDA_FLASHKDA=1 (unset/empty/0 = stock) -> the bundle overlay patch_flashkda.py: the KDA\n"
          "#   chunked prefill runs FlashKDA 17a037d (_flashkda_fp32_C, fp32 recurrent state; 2.4x the Triton chain per\n"
          "#   layer through the wrapper, 3.5x with the kda_conv quick win's contiguous q/k/v => ~0.45-0.56 s saved per\n"
          "#   13,824-token chunk; also extends the quickwins kda_conv VERIFIED fingerprint; decode untouched). On BOTH\n"
          "#   ranks: head -e after GLM53_KPOOL_DROP_LOWEST, worker serve_env_names. docs/KDA_FLASHKDA.md.\n") + K4_OLD
F_EDITS = (("F1 validation", F1_OLD, F1_NEW, 1), ("F2 head -e", F2_OLD, F2_NEW, 1), ("F3 worker list", F3_OLD, F3_NEW, 1),
           ("F4 note", F4_OLD, F4_NEW, 1))


def add_flashkda(s: str) -> tuple[str, str]:
    """r16m/r16n stage (pinned R16L_SHA) -> + F1-F4, with kpooldown's kind of checks; == committed launcher/start.sh."""
    sha = hashlib.sha256(s.encode()).hexdigest()
    if sha != R16L_SHA:
        sys.exit(f"ABORT: the r16l stage is {sha[:16]}, not {R16L_SHA[:16]} that the flashkda edits were made against")
    if re.search(rf"\b{FK_KNOB}\b", s):
        sys.exit(f"ABORT: {FK_KNOB} already in the r16l stage")
    for name, old, _new, n in F_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: flashkda {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in F_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the flashkda edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {FK_KNOB}="${{{FK_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(FK_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {FK_KNOB} "${FK_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {FK_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in F_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the flashkda stage differs from the r16l stage outside F1-F4")
    ref = REPO / "launcher" / "start.sh"
    same = ""
    if ref.is_file():
        committed = ref.read_text()
        if committed == s2:
            same = ", == committed launcher/start.sh"
        elif SP_KNOB in committed and E4_KNOB in committed:
            same = ", the committed launcher/start.sh is the later r16x stage (regenerated from it)"
        else:
            sys.exit(f"ABORT: the result differs from the committed {ref} (the reviewed r16m start.sh)")
    return s2, (f"+ flashkda F1-F4 ({FK_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}){same}")


# ---- r16x (combined kit): the mhcsp + moee4m3 knobs on top of the r16n stage (docs/MHC_SP.md, docs/MOE_E4M3.md).
# The M texts are mhcsp's, anchored on the r16n stage's OWN F results (the E texts moee4m3's, on the M results): the
# fixed chain flashkda -> mhcsp -> moee4m3 reproduces the committed launcher/start.sh, and the checks are flashkda's.
SP_KNOB = "GLM53_MHC_SP"
XM1_OLD = F1_NEW
XM1_NEW = XM1_OLD + """    # [mhcsp] sequence-parallel mHC prefill over TP (docs/MHC_SP.md), on both ranks. Unset/empty/0 = stock
    # (production byte-identical); 1 = the overlay patch_mhc_sp.py, run by patch_tf_bundle.py: a >= 1024-token
    # forward shards the residual stream (the per-token mHC family runs on T/2 rows; the attention/MLP all-reduces
    # become reduce-scatter + all-gather pairs, the same wire bytes at TP=2), decode and every CUDA-graph capture
    # stay plain TP. Anything else would stop both containers inside the bundle, so it is refused here.
    if [ -n "${GLM53_MHC_SP:-}" ]; then
        _glm53_validate_bool_flag GLM53_MHC_SP "$GLM53_MHC_SP" || return
    fi
    if [ "${GLM53_MHC_SP:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_mhc_sp.py" ]; then
            echo "GLM53_MHC_SP=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_mhc_sp.py" >&2
            return 2
        fi
        if ! grep -qF '("patch_mhc_sp.py", "GLM53_MHC_SP")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MHC_SP=1 requires $TF_BUNDLE_PATCH_HOST to run patch_mhc_sp.py (the r16msp patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
XM2_OLD = F2_NEW
XM2_NEW = XM2_OLD + '        -e GLM53_MHC_SP="${GLM53_MHC_SP:-}" \\\n'
XM3_OLD = F3_NEW
XM3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP; do  # [tf-exl3-fork]\n")
XM4_OLD = F4_NEW
XM4_NEW = ("# [mhcsp] GLM53_MHC_SP=1 (unset/empty/0 = stock) -> the bundle overlay patch_mhc_sp.py: sequence-parallel\n"
           "#   prefill for the mHC bookkeeping over TP (>= 1024-token forwards shard the residual stream; the mHC\n"
           "#   family runs on T/2 rows and the attention/MLP all-reduces become RS+AG pairs, the same wire bytes at\n"
           "#   TP=2; decode and every CUDA-graph capture stay plain TP). On BOTH ranks: head -e after\n"
           "#   GLM53_KDA_FLASHKDA, worker serve_env_names. docs/MHC_SP.md.\n") + XM4_OLD
XM_EDITS = (("M1 validation", XM1_OLD, XM1_NEW, 1), ("M2 head -e", XM2_OLD, XM2_NEW, 1),
            ("M3 worker list", XM3_OLD, XM3_NEW, 1), ("M4 note", XM4_OLD, XM4_NEW, 1))


def add_mhcsp(s: str) -> tuple[str, str]:
    """the r16n stage (flashkda F1-F4 applied) -> + M1-M4, with flashkda's kind of checks."""
    if re.search(rf"\b{SP_KNOB}\b", s):
        sys.exit(f"ABORT: {SP_KNOB} already in the stage")
    for name, old, _new, n in XM_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: mhcsp {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in XM_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the mhcsp edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {SP_KNOB}="${{{SP_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(SP_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {SP_KNOB} "${SP_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {SP_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in XM_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the mhcsp stage differs from the incoming stage outside M1-M4")
    ref = REPO / "launcher" / "start.sh"
    same = ""
    if ref.is_file():
        committed = ref.read_text()
        if committed == s2:
            same = ", == committed launcher/start.sh"
        elif E4_KNOB in committed and SP_KNOB in committed:
            same = ", the committed launcher/start.sh is the later r16x stage (regenerated from it)"
        else:
            same = ", the committed launcher/start.sh is an earlier stage (r16n: update it with this output)"
    return s2, (f"+ mhcsp M1-M4 ({SP_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}){same}")


E4_KNOB = "GLM53_MOE_E4M3"
XE1_OLD = XM1_NEW
XE1_NEW = XE1_OLD + """    # [moee4m3] the e4m3 routed-MoE prefill (docs/MOE_E4M3.md), on both ranks. Unset/empty/0 = stock (the bundle
    # does not even run the overlay: nothing of the feature reaches site-packages); 1 = overlay/patch_moe_e4m3.py,
    # run by patch_tf_bundle.py (copies glm53_moe_e4m3.py + its extension from the bundle overlay dir into
    # site-packages and arms integrate.plugin_register). Anything else would stop both containers inside the
    # bundle, so it is refused here.
    if [ -n "${GLM53_MOE_E4M3:-}" ]; then
        _glm53_validate_bool_flag GLM53_MOE_E4M3 "$GLM53_MOE_E4M3" || return
    fi
    if [ "${GLM53_MOE_E4M3:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_moe_e4m3.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so" ]; then
            echo "GLM53_MOE_E4M3=1 requires $TF_BUNDLE_DIR_HOST/overlay/{patch_moe_e4m3.py,glm53_moe_e4m3.py,glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so}" >&2
            return 2
        fi
        if ! grep -qF '("patch_moe_e4m3.py", "GLM53_MOE_E4M3")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MOE_E4M3=1 requires $TF_BUNDLE_PATCH_HOST to run patch_moe_e4m3.py (the r16e4 patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
XE2_OLD = XM2_NEW
XE2_NEW = XE2_OLD + '        -e GLM53_MOE_E4M3="${GLM53_MOE_E4M3:-}" \\\n'
XE3_OLD = XM3_NEW
XE3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3; do  # [tf-exl3-fork]\n")
XE4_OLD = XM4_NEW
XE4_NEW = ("# [moee4m3] GLM53_MOE_E4M3=1 (unset/empty/0 = stock) -> the bundle overlay patch_moe_e4m3.py: every routed-MoE\n"
           "#   apply call with more tokens than the fused cap (prefill chunks, incl. decode/verify tokens batched into\n"
           "#   them) runs on the e4m3 tensor-core kernels; decode-only steps untouched. Per-layer self-test at model\n"
           "#   load. On BOTH ranks: head -e after GLM53_MHC_SP, worker serve_env_names. docs/MOE_E4M3.md.\n"
           ) + XE4_OLD
XE_EDITS = (("E1 validation", XE1_OLD, XE1_NEW, 1), ("E2 head -e", XE2_OLD, XE2_NEW, 1),
            ("E3 worker list", XE3_OLD, XE3_NEW, 1), ("E4 note", XE4_OLD, XE4_NEW, 1))


def add_moee4m3(s: str) -> tuple[str, str]:
    """the mhcsp stage -> + E1-E4, with flashkda's kind of checks; == committed launcher/start.sh once r16x is committed."""
    if re.search(rf"\b{E4_KNOB}\b", s):
        sys.exit(f"ABORT: {E4_KNOB} already in the stage")
    for name, old, _new, n in XE_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moee4m3 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in XE_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moee4m3 edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {E4_KNOB}="${{{E4_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(E4_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {E4_KNOB} "${E4_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {E4_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in XE_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moee4m3 stage differs from the incoming stage outside E1-E4")
    ref = REPO / "launcher" / "start.sh"
    same = ""
    if ref.is_file():
        if ref.read_text() == s2:
            same = ", == committed launcher/start.sh"
        else:
            same = ", the committed launcher/start.sh is an earlier stage (update it with this output)"
    return s2, (f"+ moee4m3 E1-E4 ({E4_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}){same}")



# ---- r16y: the W8A8 prefill dense GEMM switch (docs/DENSE_W8A8.md) on top of the r16x stage. The W1-W4 texts are
# the w8a8 branch's (reviewed); the anchors are this chain's E results, and the YA4 note goes ABOVE the e4m3 note
# (the chain's notes read newest-last, like every stage before it).
WA_KNOB = "GLM53_DENSE_W8A8"
YA1_OLD = XE1_NEW
YA1_NEW = YA1_OLD + """    # [w8a8] the W8A8 prefill dense GEMMs (docs/DENSE_W8A8.md), on both ranks. Unset/empty/0 = stock (decode keeps
    # Marlin either way); 1 = the bundle overlay patch_dense_w8a8.py installs fp8_w8a8.py + tf_fp8_w8a8_ext into
    # site-packages and arms integrate.plugin_register to import them with the same env. Anything else would stop
    # both containers inside the bundle, so it is refused here.
    if [ -n "${GLM53_DENSE_W8A8:-}" ]; then
        _glm53_validate_bool_flag GLM53_DENSE_W8A8 "$GLM53_DENSE_W8A8" || return
    fi
    if [ "${GLM53_DENSE_W8A8:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_dense_w8a8.py" ]; then
            echo "GLM53_DENSE_W8A8=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_dense_w8a8.py" >&2
            return 2
        fi
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" ] || \\
           [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so" ]; then
            echo "GLM53_DENSE_W8A8=1 requires $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py and tf_fp8_w8a8_ext*.so" >&2
            return 2
        fi
        if ! grep -qF '("patch_dense_w8a8.py", "GLM53_DENSE_W8A8")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8=1 requires $TF_BUNDLE_PATCH_HOST to run patch_dense_w8a8.py (the r16y patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
YA2_OLD = XE2_NEW
YA2_NEW = YA2_OLD + '        -e GLM53_DENSE_W8A8="${GLM53_DENSE_W8A8:-}" \\\n'
YA3_OLD = XE3_NEW
YA3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8; do  # [tf-exl3-fork]\n")
YA4_OLD = XE4_NEW
YA4_NEW = ("# [w8a8] GLM53_DENSE_W8A8=1 (unset/empty/0 = stock) -> the bundle overlay patch_dense_w8a8.py installs\n"
           "#   fp8_w8a8.py + tf_fp8_w8a8_ext into site-packages and arms the site integrate.py on BOTH ranks (head -e\n"
           "#   after GLM53_MOE_E4M3, worker serve_env_names): per-token fp8 activations x the stored per-channel fp8\n"
           "#   weights (cutlass_scaled_mm) for the dense + shared-expert FP8 linears on the PREFILL path only; the\n"
           "#   standard fp8 operand is repacked per call from the Marlin payload, no resident copy (KV pool 2.00M ->\n"
           "#   1.96M). Decode keeps Marlin. docs/DENSE_W8A8.md.\n"
           ) + YA4_OLD
YA_EDITS = (("YA1 validation", YA1_OLD, YA1_NEW, 1), ("YA2 head -e", YA2_OLD, YA2_NEW, 1),
            ("YA3 worker list", YA3_OLD, YA3_NEW, 1), ("YA4 note", YA4_OLD, YA4_NEW, 1))


def add_dense_w8a8(s: str) -> tuple[str, str]:
    """the r16x stage -> + YA1-YA4, with flashkda's kind of checks; == committed launcher/start.sh once r16y is committed."""
    if re.search(rf"\b{WA_KNOB}\b", s):
        sys.exit(f"ABORT: {WA_KNOB} already in the stage")
    for name, old, _new, n in YA_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: w8a8 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in YA_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the w8a8 edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {WA_KNOB}="${{{WA_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(WA_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {WA_KNOB} "${WA_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {WA_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in YA_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the w8a8 stage differs from the incoming stage outside YA1-YA4")
    ref = REPO / "launcher" / "start.sh"
    same = ""
    if ref.is_file():
        if ref.read_text() == s2:
            same = ", == committed launcher/start.sh"
        else:
            same = ", the committed launcher/start.sh is an earlier stage (update it with this output)"
    return s2, (f"+ w8a8 YA1-YA4 ({WA_KNOB}: head -e x{head}, worker list x{worker}, validation x{val}){same}")


# ---- r16z: the FlashKDA BUILD switch (docs/KDA_FLASHKDA3.md) and the pipelined-SP switch (docs/MHC_SP2.md) on top
# of the r16y stage. The ZV/ZS texts are anchored on the YA results (the chain continued: F -> M -> E -> YA -> ZV
# -> ZS); both new switches default to UNSET = production byte for byte, like every stage before them.
FV_KNOB = "GLM53_KDA_FLASHKDA_V"
ZV1_OLD = YA1_NEW
ZV1_NEW = ZV1_OLD + """    # [fkda-v] WHICH FlashKDA build GLM53_KDA_FLASHKDA=1 installs (docs/KDA_FLASHKDA3.md), on both ranks.
    # Unset/empty/1 = the shipped r16x build (production's bytes today: overlay/{glm53_flashkda.py,
    # _flashkda_fp32_C.abi3.so}); 2 = the fkda2 precision build; 3 = the fkda3 build (direct-output kda.py).
    # All three stage under the SAME site-packages names and each wrapper pins its extension sha at boot; the
    # value is only read when GLM53_KDA_FLASHKDA=1. Anything else would stop both containers inside the bundle,
    # so it is refused here (and the =2 / =3 overlay files are checked before a rank is stopped).
    if [ -n "${GLM53_KDA_FLASHKDA_V:-}" ]; then
        case "$GLM53_KDA_FLASHKDA_V" in
            1|2|3) ;;
            *) echo "GLM53_KDA_FLASHKDA_V must be unset, 1, 2 or 3 (value not printed)" >&2; return 2;;
        esac
        if [ "$GLM53_KDA_FLASHKDA_V" = "2" ]; then
            if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_flashkda2.py" ] \\
               || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/_flashkda_fp32_C2.abi3.so" ]; then
                echo "GLM53_KDA_FLASHKDA_V=2 requires $TF_BUNDLE_DIR_HOST/overlay/glm53_flashkda2.py and _flashkda_fp32_C2.abi3.so" >&2
                return 2
            fi
        fi
        if [ "$GLM53_KDA_FLASHKDA_V" = "3" ]; then
            if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_flashkda3.py" ] \\
               || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/_flashkda_fp32_C3.abi3.so" ]; then
                echo "GLM53_KDA_FLASHKDA_V=3 requires $TF_BUNDLE_DIR_HOST/overlay/glm53_flashkda3.py and _flashkda_fp32_C3.abi3.so" >&2
                return 2
            fi
        fi
        if ! grep -qF '("patch_flashkda.py", "GLM53_KDA_FLASHKDA")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_KDA_FLASHKDA_V requires $TF_BUNDLE_PATCH_HOST to run patch_flashkda.py (the r16m patch_tf_bundle.py; the value is only read there)" >&2
            return 2
        fi
        if ! grep -qF "GLM53_KDA_FLASHKDA_V" "$TF_BUNDLE_DIR_HOST/overlay/patch_flashkda.py" 2>/dev/null; then
            echo "GLM53_KDA_FLASHKDA_V requires the r16z overlay patch_flashkda.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/patch_flashkda.py is an older one that would silently stage the shipped build" >&2
            return 2
        fi
    fi
"""
ZV2_OLD = YA2_NEW
ZV2_NEW = ZV2_OLD + '        -e GLM53_KDA_FLASHKDA_V="${GLM53_KDA_FLASHKDA_V:-}" \\\n'
ZV3_OLD = YA3_NEW
ZV3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V; do  # [tf-exl3-fork]\n")
ZV4_OLD = YA4_NEW
ZV4_NEW = ("# [fkda-v] GLM53_KDA_FLASHKDA_V (unset/1 = the shipped r16x FlashKDA build = production bytes; 2 = the fkda2\n"
           "#   precision build; 3 = the fkda3 build) picks WHICH FlashKDA GLM53_KDA_FLASHKDA=1 stages into\n"
           "#   site-packages (all three under the same names glm53_flashkda.py / _flashkda_fp32_C.abi3.so; each wrapper\n"
           "#   pins its extension sha at boot and the fkda3 kda.py call passes out= so FlashKDA writes the layer output\n"
           "#   directly). Switching builds on an installed tree is refused (byte mismatch): FK=0 restart, then FK=1 with\n"
           "#   the new value. docs/KDA_FLASHKDA3.md.\n") + ZV4_OLD
ZV_EDITS = (("ZV1 validation", ZV1_OLD, ZV1_NEW, 1), ("ZV2 head -e", ZV2_OLD, ZV2_NEW, 1),
            ("ZV3 worker list", ZV3_OLD, ZV3_NEW, 1), ("ZV4 note", ZV4_OLD, ZV4_NEW, 1))


def add_fkda_version(s: str) -> tuple[str, str]:
    """the r16y stage -> + ZV1-ZV4 (the FlashKDA BUILD switch), with flashkda's kind of checks."""
    if re.search(rf"\b{FV_KNOB}\b", s):
        sys.exit(f"ABORT: {FV_KNOB} already in the stage")
    for name, old, _new, n in ZV_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: fkda-v {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZV_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the fkda-v edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {FV_KNOB}="${{{FV_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(FV_KNOB) if m else 0
    val = s2.count('"GLM53_KDA_FLASHKDA_V must be unset, 1, 2 or 3')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {FV_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZV_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the fkda-v stage differs from the incoming stage outside ZV1-ZV4")
    return s2, (f"+ fkda-v ZV1-ZV4 ({FV_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")


SP2_KNOB = "GLM53_MHC_SP2"
ZS1_OLD = ZV1_NEW
ZS1_NEW = ZS1_OLD + """    # [mhcsp2] the PIPELINED SP prefill (docs/MHC_SP2.md), on both ranks. Unset/empty/0 = the r16x SP result
    # (GLM53_MHC_SP alone); 1 = the overlay patch_mhc_sp2.py, run by patch_tf_bundle.py AFTER patch_mhc_sp.py:
    # k=2 interleaved sub-chunks per rank shard with the reduce-scatter / all-gather of each sub-chunk on a side
    # stream, and odd-T SP. REQUIRES GLM53_MHC_SP=1 on BOTH ranks (the pipelined SP extends the r16x SP patch of
    # the same model.py; the paired-collective hazard of mhcsp D-A, so it is refused here as well and the patch
    # refuses inside the container).
    if [ -n "${GLM53_MHC_SP2:-}" ]; then
        _glm53_validate_bool_flag GLM53_MHC_SP2 "$GLM53_MHC_SP2" || return
    fi
    if [ "${GLM53_MHC_SP2:-}" = "1" ]; then
        if [ "${GLM53_MHC_SP:-}" != "1" ]; then
            echo "GLM53_MHC_SP2=1 requires GLM53_MHC_SP=1 (the pipelined SP extends the r16x SP patch)" >&2
            return 2
        fi
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_mhc_sp2.py" ]; then
            echo "GLM53_MHC_SP2=1 requires $TF_BUNDLE_DIR_HOST/overlay/patch_mhc_sp2.py" >&2
            return 2
        fi
        if ! grep -qF '("patch_mhc_sp2.py", "GLM53_MHC_SP2")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MHC_SP2=1 requires $TF_BUNDLE_PATCH_HOST to run patch_mhc_sp2.py (the r16z patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
ZS2_OLD = ZV2_NEW
ZS2_NEW = ZS2_OLD + '        -e GLM53_MHC_SP2="${GLM53_MHC_SP2:-}" \\\n'
ZS3_OLD = ZV3_NEW
ZS3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2; do  "
           "# [tf-exl3-fork]\n")
ZS4_OLD = ZV4_NEW
ZS4_NEW = ("# [mhcsp2] GLM53_MHC_SP2=1 (unset/empty/0 = the r16x SP result) -> the bundle overlay patch_mhc_sp2.py: the\n"
           "#   SP prefill runs k=2 interleaved sub-chunks per rank shard, the reduce-scatter / all-gather of each\n"
           "#   sub-chunk on a side stream overlapping the neighbouring sub-chunk's mHC (bitwise the r16x SP result at\n"
           "#   sub-chunks >= 1,537 rows), and odd token counts are sharded too. NEEDS GLM53_MHC_SP=1 (both ranks).\n"
           "#   docs/MHC_SP2.md.\n") + ZS4_OLD
ZS_EDITS = (("ZS1 validation", ZS1_OLD, ZS1_NEW, 1), ("ZS2 head -e", ZS2_OLD, ZS2_NEW, 1),
            ("ZS3 worker list", ZS3_OLD, ZS3_NEW, 1), ("ZS4 note", ZS4_OLD, ZS4_NEW, 1))


def add_mhcsp2(s: str) -> tuple[str, str]:
    """the fkda-v stage -> + ZS1-ZS4 (the pipelined-SP switch), with flashkda's kind of checks."""
    if re.search(rf"\b{SP2_KNOB}\b", s):
        sys.exit(f"ABORT: {SP2_KNOB} already in the stage")
    for name, old, _new, n in ZS_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: mhcsp2 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZS_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the mhcsp2 edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {SP2_KNOB}="${{{SP2_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(SP2_KNOB) if m else 0
    val = s2.count('_glm53_validate_bool_flag GLM53_MHC_SP2 "$GLM53_MHC_SP2"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {SP2_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZS_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the mhcsp2 stage differs from the incoming stage outside ZS1-ZS4")
    return s2, (f"+ mhcsp2 ZS1-ZS4 ({SP2_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")

# ---- r16z2: the w8a82 knobs (docs/DENSE_W8A8_2.md) and the e4m3 down-projection width (docs/MOE2.md) on top of the
# r16z stage. The ZW/ZD texts are anchored on the ZS results (the chain continued: F -> M -> E -> YA -> ZV -> ZS ->
# ZW -> ZD); all three knobs default to UNSET = production byte for byte and are only read by their parent switch's
# module (GLM53_DENSE_W8A8_GEMM / _ONLY by overlay/fp8_w8a8.py when GLM53_DENSE_W8A8=1, GLM53_MOE_E4M3_DOWN by
# overlay/glm53_moe_e4m3.py when GLM53_MOE_E4M3=1).
WG_KNOB = "GLM53_DENSE_W8A8_GEMM"
WO_KNOB = "GLM53_DENSE_W8A8_ONLY"
ZW2_OLD = ZS2_NEW
ZW3_OLD = ZS3_NEW
ZW4_OLD = ZS4_NEW
ZW1_OLD = ZS1_NEW
ZW1_NEW = ZW1_OLD + """    # [w8a82] the W8A8 GEMM backend and the projection filter (docs/DENSE_W8A8_2.md), on both ranks. Only
    # read when GLM53_DENSE_W8A8=1. GLM53_DENSE_W8A8_GEMM: unset/empty = custom (this extension's CUTLASS SM120
    # persistent GEMM, epilogue arithmetic identical to cutlass_scaled_mm and BITWISE == it, checked per shape at
    # load; a shape whose check fails falls back to cutlass_mm on its own); cutlass_mm = the image's
    # cutlass_scaled_mm in pieces (the w8a8 path). GLM53_DENSE_W8A8_ONLY: unset = every served projection; else a
    # comma list of "<group>.<projection>" from the module's PROJ_NAMES (the production A/B dial: e.g.
    # kda.in_proj_qkvbfg_a,mla.o_proj carries ~65 % of the saving through 45 of the 192 GEMMs per chunk). Anything
    # else would stop both containers inside the bundle, so it is refused here - and both knobs need the r16z2
    # overlay fp8_w8a8.py that reads them (an older one would silently ignore the value).
    if [ -n "${GLM53_DENSE_W8A8_GEMM:-}" ]; then
        case "$GLM53_DENSE_W8A8_GEMM" in
            custom|cutlass_mm) ;;
            *) echo "GLM53_DENSE_W8A8_GEMM must be unset, custom or cutlass_mm (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_DENSE_W8A8_GEMM" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_GEMM requires the r16z2 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently serve the cutlass_mm path" >&2
            return 2
        fi
    fi
    if [ -n "${GLM53_DENSE_W8A8_ONLY:-}" ]; then
        __w82_it="${GLM53_DENSE_W8A8_ONLY},"
        while [ -n "$__w82_it" ]; do
            __w82_p="${__w82_it%%,*}"; __w82_it="${__w82_it#*,}"
            case "$__w82_p" in
                ""|kda.in_proj_qkvbfg_a|kda.o_proj|mla.fused_qkv_a_proj|mla.q_b_proj|mla.o_proj|shared.gate_up_proj|shared.down_proj|dense.gate_up_proj|dense.down_proj) ;;
                *) echo "GLM53_DENSE_W8A8_ONLY: a name that is not one of the module's PROJ_NAMES (value not printed)" >&2; return 2;;
            esac
        done
        if ! grep -qF "GLM53_DENSE_W8A8_ONLY" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_ONLY requires the r16z2 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently serve every projection" >&2
            return 2
        fi
    fi
"""
ZW2_NEW = ZW2_OLD + '        -e GLM53_DENSE_W8A8_GEMM="${GLM53_DENSE_W8A8_GEMM:-}" \\\n' \
                   '        -e GLM53_DENSE_W8A8_ONLY="${GLM53_DENSE_W8A8_ONLY:-}" \\\n'
ZW3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY; do  # [tf-exl3-fork]\n")
ZW4_NEW = ("# [w8a82] GLM53_DENSE_W8A8_GEMM (unset/custom = the custom CUTLASS SM120 persistent GEMM, bitwise ==\n"
           "#   cutlass_scaled_mm, checked per shape at load, a failing shape falls back to cutlass_mm on its own;\n"
           "#   cutlass_mm = the image's cutlass_scaled_mm in pieces) and GLM53_DENSE_W8A8_ONLY (unset = every served\n"
           "#   projection; else a comma list of the module's PROJ_NAMES, e.g. kda.in_proj_qkvbfg_a,mla.o_proj) dial the\n"
           "#   W8A8 prefill dense GEMMs (only read when GLM53_DENSE_W8A8=1). docs/DENSE_W8A8_2.md.\n") + ZW4_OLD
ZW_EDITS = (("ZW1 validation", ZW1_OLD, ZW1_NEW, 1), ("ZW2 head -e", ZW2_OLD, ZW2_NEW, 1),
            ("ZW3 worker list", ZW3_OLD, ZW3_NEW, 1), ("ZW4 note", ZW4_OLD, ZW4_NEW, 1))


def add_w8a82(s: str) -> tuple[str, str]:
    """the r16z stage -> + ZW1-ZW4 (the w8a82 knobs), with flashkda's kind of checks."""
    if re.search(rf"\b{WG_KNOB}\b", s):
        sys.exit(f"ABORT: {WG_KNOB} already in the stage")
    for name, old, _new, n in ZW_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: w8a82 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZW_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the w8a82 edits: {r.stderr.decode()}")
    for k in (WG_KNOB, WO_KNOB):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count('"GLM53_DENSE_W8A8_GEMM must be unset, custom or cutlass_mm')
    if val != 1:
        sys.exit(f"ABORT: {WG_KNOB} validation x{val} (need 1)")
    back = s2
    for _name, old, new, n in ZW_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the w8a82 stage differs from the incoming stage outside ZW1-ZW4")
    return s2, (f"+ w8a82 ZW1-ZW4 ({WG_KNOB} + {WO_KNOB}: head -e x1 each, worker list x1 each, validation x1)")


MD_KNOB = "GLM53_MOE_E4M3_DOWN"
ZD2_OLD = ZW2_NEW
ZD3_OLD = ZW3_NEW
ZD4_OLD = ZW4_NEW
ZD1_OLD = ZW1_NEW
ZD1_NEW = ZD1_OLD + """    # [moe2] the e4m3 down-projection width (docs/MOE2.md), on both ranks. Only read when GLM53_MOE_E4M3=1.
    # unset/empty/e4m3 = the down projection on e4m3 too (the original spec); f16 = the fused variant 16: gate/up on
    # e4m3, the down projection on production's operand widths (fp16 rotated input x fp16 trellis decode, fp32
    # accumulate; removes 2 of the 4 e4m3 roundings for ~+1.5 ms per 13,824-token layer call). Anything else would
    # stop both containers inside the bundle, so it is refused here - and the knob needs the r16z2 overlay
    # glm53_moe_e4m3.py that reads it (an older one would silently serve e4m3).
    if [ -n "${GLM53_MOE_E4M3_DOWN:-}" ]; then
        case "$GLM53_MOE_E4M3_DOWN" in
            e4m3|f16) ;;
            *) echo "GLM53_MOE_E4M3_DOWN must be unset, e4m3 or f16 (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_MOE_E4M3_DOWN" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "GLM53_MOE_E4M3_DOWN requires the r16z2 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently serve e4m3" >&2
            return 2
        fi
    fi
"""
ZD2_NEW = ZW2_NEW + '        -e GLM53_MOE_E4M3_DOWN="${GLM53_MOE_E4M3_DOWN:-}" \\\n'
ZD3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN; do  # [tf-exl3-fork]\n")
ZD4_NEW = ("# [moe2] GLM53_MOE_E4M3_DOWN (unset/empty/e4m3 = the down projection on e4m3 too, the original spec; f16 =\n"
           "#   the fused variant 16: gate/up on e4m3, the down projection on production's fp16 operand widths, fp32\n"
           "#   accumulate - removes 2 of the 4 e4m3 roundings, docs/MOE2.md) dials the e4m3 routed-MoE prefill (only\n"
           "#   read when GLM53_MOE_E4M3=1).\n") + ZD4_OLD
ZD_EDITS = (("ZD1 validation", ZD1_OLD, ZD1_NEW, 1), ("ZD2 head -e", ZD2_OLD, ZD2_NEW, 1),
            ("ZD3 worker list", ZD3_OLD, ZD3_NEW, 1), ("ZD4 note", ZD4_OLD, ZD4_NEW, 1))


def add_moe_down(s: str) -> tuple[str, str]:
    """the w8a82 stage -> + ZD1-ZD4 (the e4m3 down width), with flashkda's kind of checks."""
    if re.search(rf"\b{MD_KNOB}\b", s):
        sys.exit(f"ABORT: {MD_KNOB} already in the stage")
    for name, old, _new, n in ZD_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moe2 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZD_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moe2 edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {MD_KNOB}="${{{MD_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(MD_KNOB) if m else 0
    val = s2.count('"GLM53_MOE_E4M3_DOWN must be unset, e4m3 or f16')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {MD_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZD_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moe2 stage differs from the incoming stage outside ZD1-ZD4")
    return s2, (f"+ moe2 ZD1-ZD4 ({MD_KNOB}: head -e x1, worker list x1, validation x1)")


# ---- r16z3 (moe3): the fused16 switch (docs/MOE3.md) and the two e4m3 layer lists on top of the r16z2 stage. The
# ZF/ZL texts are anchored on the ZD results (the chain continued: F -> M -> E -> YA -> ZV -> ZS -> ZW -> ZD -> ZF
# -> ZL); all three knobs default to UNSET = production byte for byte. GLM53_MOE_FUSED16 is read by
# overlay/glm53_moe_fused16.py (installed+armed by patch_moe_fused16.py: its integrate.py block goes BEFORE
# glm53_moe_e4m3's, which must stay last); the two layer lists are read by overlay/glm53_moe_e4m3.py and are only
# in effect when GLM53_MOE_E4M3=1.
FF_KNOB = "GLM53_MOE_FUSED16"
ZF2_OLD = ZD2_NEW
ZF3_OLD = ZD3_NEW
ZF4_OLD = ZD4_NEW
ZF1_OLD = ZD1_NEW
ZF1_NEW = ZF1_OLD + """    # [moe3] the fused16 routed-MoE prefill (docs/MOE3.md), on both ranks. Unset/empty/0 = stock (the bundle
    # does not even run the overlay: nothing of the feature reaches site-packages); 1 = overlay/patch_moe_fused16.py,
    # run by patch_tf_bundle.py (copies glm53_moe_fused16.py + the extension it shares with GLM53_MOE_E4M3 into
    # site-packages and arms integrate.py; its block sits BEFORE glm53_moe_e4m3's, which must stay last): every E3
    # grouped prefill call >= 4,096 tokens runs production's arithmetic on the P16 schedules (h2 bit-identical, out =
    # production's up to the fp32 atomic-add order), <= 4,096 / CUDA-graph capture passes through. Anything else
    # would stop both containers inside the bundle, so it is refused here.
    if [ -n "${GLM53_MOE_FUSED16:-}" ]; then
        _glm53_validate_bool_flag GLM53_MOE_FUSED16 "$GLM53_MOE_FUSED16" || return
    fi
    if [ "${GLM53_MOE_FUSED16:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_moe_fused16.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_fused16.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so" ]; then
            echo "GLM53_MOE_FUSED16=1 requires $TF_BUNDLE_DIR_HOST/overlay/{patch_moe_fused16.py,glm53_moe_fused16.py,glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so}" >&2
            return 2
        fi
        if ! grep -qF '("patch_moe_fused16.py", "GLM53_MOE_FUSED16")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MOE_FUSED16=1 requires $TF_BUNDLE_PATCH_HOST to run patch_moe_fused16.py (the r16z3 patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
ZF2_NEW = ZF2_OLD + '        -e GLM53_MOE_FUSED16="${GLM53_MOE_FUSED16:-}" \\\n'
ZF3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16; do  "
           "# [tf-exl3-fork]\n")
ZF4_NEW = ("# [moe3] GLM53_MOE_FUSED16=1 (unset/empty/0 = stock) -> the bundle overlay patch_moe_fused16.py: every E3\n"
           "#   grouped prefill call with more tokens than 4,096 runs production's arithmetic in the P16 schedules\n"
           "#   (persistent 2-CTA p16b, >= 11,264 tokens; sep, a side stream per segment chunk; production's own kernels\n"
           "#   below). h2 bit-identical, out = production's up to the fp32 atomic-add order; per-layer self-test at\n"
           "#   model load. On BOTH ranks: head -e after GLM53_MOE_E4M3_DOWN, worker serve_env_names. docs/MOE3.md.\n"
           ) + ZF4_OLD
ZF_EDITS = (("ZF1 validation", ZF1_OLD, ZF1_NEW, 1), ("ZF2 head -e", ZF2_OLD, ZF2_NEW, 1),
            ("ZF3 worker list", ZF3_OLD, ZF3_NEW, 1), ("ZF4 note", ZF4_OLD, ZF4_NEW, 1))


def add_moe_fused16(s: str) -> tuple[str, str]:
    """the moe2 stage -> + ZF1-ZF4 (the fused16 switch), with flashkda's kind of checks."""
    if re.search(rf"\b{FF_KNOB}\b", s):
        sys.exit(f"ABORT: {FF_KNOB} already in the stage")
    for name, old, _new, n in ZF_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moe3 {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZF_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moe3 edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {FF_KNOB}="${{{FF_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(FF_KNOB) if m else 0
    val = s2.count(f'_glm53_validate_bool_flag {FF_KNOB} "${FF_KNOB}"')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {FF_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZF_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moe3 stage differs from the incoming stage outside ZF1-ZF4")
    return s2, (f"+ moe3 ZF1-ZF4 ({FF_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")


LS_KNOB = "GLM53_MOE_E4M3_LAYERS"
LD_KNOB = "GLM53_MOE_E4M3_DOWN_LAYERS"
ZL2_OLD = ZF2_NEW
ZL3_OLD = ZF3_NEW
ZL4_OLD = ZF4_NEW
ZL1_OLD = ZF1_NEW
# the layer-list validation (shared text for both knobs, parameterised by name): comma list / ranges, every index an
# integer inside the model's MoE layers 3..44 (the RoutedExperts layer_name "model.layers.<i>.mlp.experts" band the
# module parses, glm53_moe_e4m3.py layer_index); whitespace is allowed around the parts (the module strips it). Only
# read when GLM53_MOE_E4M3=1; the value needs the r16z3 overlay glm53_moe_e4m3.py that reads it (an older one would
# silently serve every layer).
ZL_VALIDATE = """    if [ -n "${{KNOB}:-}" ]; then
        __moe3_it="${{KNOB}},"
        while [ -n "$__moe3_it" ]; do
            __moe3_p="${__moe3_it%%,*}"; __moe3_it="${__moe3_it#*,}"
            __moe3_p=$(printf '%s' "$__moe3_p" | tr -d '[:space:]')
            case "$__moe3_p" in
                "") ;;
                *[!0-9-]*) echo "{KNOB}: a part that is not a layer index or range (value not printed)" >&2; return 2;;
                *-*) case "${__moe3_p%%-*}${__moe3_p#*-}" in *[!0-9]*) echo "{KNOB}: a malformed range (value not printed)" >&2; return 2;; esac
                     [ "${__moe3_p%%-*}" -le "${__moe3_p#*-}" ] || { echo "{KNOB}: a descending range (value not printed)" >&2; return 2; }
                     [ "${{KNOB}:+x}" ] && { [ "${__moe3_p%%-*}" -ge {LO} ] && [ "${__moe3_p#*-}" -le {HI} ] || { echo "{KNOB}: a range outside the model's MoE layers {LO}..{HI}" >&2; return 2; }; };;
                *) [ "${{KNOB}:+x}" ] && { [ "$__moe3_p" -ge {LO} ] && [ "$__moe3_p" -le {HI} ] || { echo "{KNOB}: an index outside the model's MoE layers {LO}..{HI}" >&2; return 2; }; };;
            esac
        done
        if ! grep -qF "{KNOB}" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "{KNOB} requires the r16z3 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently serve every layer" >&2
            return 2
        fi
    fi
"""
ZL1_NEW = ZL1_OLD + ZL_VALIDATE.replace("{KNOB}", LS_KNOB).replace("{LO}", "3").replace("{HI}", "44") + \
    ZL_VALIDATE.replace("{KNOB}", LD_KNOB).replace("{LO}", "3").replace("{HI}", "44")
ZL2_NEW = ZL2_OLD + f'        -e {LS_KNOB}="${{{LS_KNOB}:-}}" \\\n' \
                    f'        -e {LD_KNOB}="${{{LD_KNOB}:-}}" \\\n'
ZL3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16 "
           "GLM53_MOE_E4M3_LAYERS GLM53_MOE_E4M3_DOWN_LAYERS; do  # [tf-exl3-fork]\n")
ZL4_NEW = ("# [moe3] GLM53_MOE_E4M3_LAYERS / GLM53_MOE_E4M3_DOWN_LAYERS (comma lists / ranges of the model's MoE layer\n"
           "#   indices 3..44; unset = every MoE layer / GLM53_MOE_E4M3_DOWN decides, unchanged) pick WHICH layers run\n"
           "#   the e4m3 path (the other layers stay on production's kernels or the P16 ones of GLM53_MOE_FUSED16) and\n"
           "#   which layers run the f16 down projection. Only read when GLM53_MOE_E4M3=1; the e4m3 summary line names\n"
           "#   the selection (\"K indices selected, U layers unselected\"). docs/MOE3.md.\n") + ZL4_OLD
ZL_EDITS = (("ZL1 validation", ZL1_OLD, ZL1_NEW, 1), ("ZL2 head -e", ZL2_OLD, ZL2_NEW, 1),
            ("ZL3 worker list", ZL3_OLD, ZL3_NEW, 1), ("ZL4 note", ZL4_OLD, ZL4_NEW, 1))


def add_moe_layers(s: str) -> tuple[str, str]:
    """the fused16 stage -> + ZL1-ZL4 (the two e4m3 layer lists), with flashkda's kind of checks."""
    for k in (LS_KNOB, LD_KNOB):
        if re.search(rf"\b{k}\b", s):
            sys.exit(f"ABORT: {k} already in the stage")
    for name, old, _new, n in ZL_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moe3 layers {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZL_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moe3 layers edits: {r.stderr.decode()}")
    for k in (LS_KNOB, LD_KNOB):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        val = s2.count(f'"{k}: a part that is not a layer index or range')
        if (head, worker, val) != (1, 1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZL_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moe3 layers stage differs from the incoming stage outside ZL1-ZL4")
    return s2, (f"+ moe3 layers ZL1-ZL4 ({LS_KNOB} + {LD_KNOB}: head -e x1 each, worker list x1 each, validation x1 each)")


# ---- r16z4: the opt-moe dials of the e4m3 path (docs/OPT_MOE.md), the MLA prefill fused index pass
# (docs/OPT_DENSE.md), the fp8 sequence-parallel all-gather and the hi+lo W8A8 knobs (docs/OPT_DENSE.md), on top of
# the r16z3 stage. The ZT/ZP/ZG/ZH texts are anchored on the ZL results (the chain continued: F -> M -> E -> YA ->
# ZV -> ZS -> ZW -> ZD -> ZF -> ZL -> ZT -> ZP -> ZG -> ZH); all seven knobs default to UNSET = production byte for
# byte (TOKGATHER's UNSET is the designed ON - bitwise the per-pair result; ACC and FOLD default OFF: the MoE
# refutation of opt-moe is still running, a rejection arrives as a follow-up; FP8AG / HILO / HILO_SEL default OFF).
# ACC / FOLD_SHARED / TOKGATHER are read by overlay/glm53_moe_e4m3.py (only when GLM53_MOE_E4M3=1), FUSED_INDEX by
# the site glm53_mla_prefill.py this kit ships (only when GLM53_MLA_PREFILL=1), FP8AG / HILO / HILO_SEL by
# overlay/fp8_w8a8.py (only when GLM53_DENSE_W8A8=1). FOLD_SHARED=1 without ACC=bf16 is refused HERE (the module
# would refuse the whole e4m3 install - production's e4m3 =1 must never lose its kernels to a dial).
AO_KNOB = "GLM53_MOE_E4M3_ACC"
FO_KNOB = "GLM53_MOE_E4M3_FOLD_SHARED"
TG_KNOB = "GLM53_MOE_E4M3_TOKGATHER"
ZT2_OLD = ZL2_NEW
ZT3_OLD = ZL3_NEW
ZT4_OLD = ZL4_NEW
ZT1_OLD = ZL1_NEW
ZT1_NEW = ZT1_OLD + """    # [moe-opt] the opt-moe dials of the e4m3 routed-MoE prefill (docs/OPT_MOE.md), on both ranks. Only read when
    # GLM53_MOE_E4M3=1. GLM53_MOE_E4M3_ACC: unset/empty/f32 = the fp32 accumulator (production's arithmetic);
    # bf16 = the bf16 accumulator (the down epilogue adds bf16-rounded contributions with red.add.noftz.v4.bf16x2,
    # -7.0 ms per 13,824-token layer call). GLM53_MOE_E4M3_FOLD_SHARED: unset/0 = unchanged; 1 = the routed sum is
    # accumulated straight into the shared experts' bf16 output (-2.05 ms) - REQUIRES GLM53_MOE_E4M3_ACC=bf16,
    # refused here BEFORE any rank is stopped (the module would refuse the whole e4m3 install).
    # GLM53_MOE_E4M3_TOKGATHER: unset/1 = on (the designed default: one gathered gate/up row per token when the
    # layer's experts share the w13 suh - bitwise the per-pair result); 0 = the per-pair gather. Anything else
    # would stop both containers inside the bundle, so it is refused here - and all three need the r16z4 overlay
    # glm53_moe_e4m3.py that reads them (an older one would silently serve fp32 / unfolded / per-pair).
    if [ -n "${GLM53_MOE_E4M3_ACC:-}" ]; then
        case "$GLM53_MOE_E4M3_ACC" in
            f32|bf16) ;;
            *) echo "GLM53_MOE_E4M3_ACC must be unset, f32 or bf16 (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_MOE_E4M3_ACC" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "GLM53_MOE_E4M3_ACC requires the r16z4 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently serve the fp32 accumulator" >&2
            return 2
        fi
    fi
    if [ -n "${GLM53_MOE_E4M3_FOLD_SHARED:-}" ]; then
        case "$GLM53_MOE_E4M3_FOLD_SHARED" in
            0|1) ;;
            *) echo "GLM53_MOE_E4M3_FOLD_SHARED must be unset, 0 or 1 (value not printed)" >&2; return 2;;
        esac
        if [ "$GLM53_MOE_E4M3_FOLD_SHARED" = 1 ] && [ "${GLM53_MOE_E4M3_ACC:-}" != bf16 ]; then
            echo "GLM53_MOE_E4M3_FOLD_SHARED=1 requires GLM53_MOE_E4M3_ACC=bf16 (the routed sum accumulates into the shared experts' bf16 output; without it the module refuses the whole e4m3 install)" >&2
            return 2
        fi
        if ! grep -qF "GLM53_MOE_E4M3_FOLD_SHARED" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "GLM53_MOE_E4M3_FOLD_SHARED requires the r16z4 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently run the unfolded path" >&2
            return 2
        fi
    fi
    if [ -n "${GLM53_MOE_E4M3_TOKGATHER:-}" ]; then
        case "$GLM53_MOE_E4M3_TOKGATHER" in
            0|1) ;;
            *) echo "GLM53_MOE_E4M3_TOKGATHER must be unset, 0 or 1 (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_MOE_E4M3_TOKGATHER" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "GLM53_MOE_E4M3_TOKGATHER requires the r16z4 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently serve the per-pair gather" >&2
            return 2
        fi
    fi
"""
ZT2_NEW = ZT2_OLD + '        -e GLM53_MOE_E4M3_ACC="${GLM53_MOE_E4M3_ACC:-}" \\\n' \
                    '        -e GLM53_MOE_E4M3_FOLD_SHARED="${GLM53_MOE_E4M3_FOLD_SHARED:-}" \\\n' \
                    '        -e GLM53_MOE_E4M3_TOKGATHER="${GLM53_MOE_E4M3_TOKGATHER:-}" \\\n'
ZT3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16 "
           "GLM53_MOE_E4M3_LAYERS GLM53_MOE_E4M3_DOWN_LAYERS GLM53_MOE_E4M3_ACC GLM53_MOE_E4M3_FOLD_SHARED "
           "GLM53_MOE_E4M3_TOKGATHER; do  # [tf-exl3-fork]\n")
ZT4_NEW = ("# [moe-opt] GLM53_MOE_E4M3_ACC (unset/f32 = the fp32 accumulator, production's arithmetic; bf16 = the bf16\n"
           "#   accumulator, -7.0 ms per 13,824-token layer call), GLM53_MOE_E4M3_FOLD_SHARED (=1 accumulates the\n"
           "#   routed sum straight into the shared experts' bf16 output, -2.05 ms; REQUIRES ACC=bf16) and\n"
           "#   GLM53_MOE_E4M3_TOKGATHER (unset/1 = the token gather, the designed default, bitwise the per-pair\n"
           "#   result; 0 = the per-pair gather) dial the e4m3 routed-MoE prefill (only read when GLM53_MOE_E4M3=1).\n"
           "#   ACC and FOLD default OFF (the MoE refutation is still running). docs/OPT_MOE.md.\n") + ZT4_OLD
ZT_EDITS = (("ZT1 validation", ZT1_OLD, ZT1_NEW, 1), ("ZT2 head -e", ZT2_OLD, ZT2_NEW, 1),
            ("ZT3 worker list", ZT3_OLD, ZT3_NEW, 1), ("ZT4 note", ZT4_OLD, ZT4_NEW, 1))


def add_moe_opt(s: str) -> tuple[str, str]:
    """the moe3 layers stage -> + ZT1-ZT4 (the opt-moe dials), with flashkda's kind of checks."""
    for k in (AO_KNOB, FO_KNOB, TG_KNOB):
        if re.search(rf"\b{k}\b", s):
            sys.exit(f"ABORT: {k} already in the stage")
    for name, old, _new, n in ZT_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moe-opt {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZT_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moe-opt edits: {r.stderr.decode()}")
    for k in (AO_KNOB, FO_KNOB, TG_KNOB):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        val = s2.count(f'"{k} must be unset')
        if (head, worker, val) != (1, 1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    val2 = s2.count('GLM53_MOE_E4M3_FOLD_SHARED=1 requires GLM53_MOE_E4M3_ACC=bf16')
    if val2 != 1:
        sys.exit(f"ABORT: the FOLD_SHARED-needs-ACC check x{val2} (need 1)")
    back = s2
    for _name, old, new, n in ZT_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moe-opt stage differs from the incoming stage outside ZT1-ZT4")
    return s2, (f"+ moe-opt ZT1-ZT4 ({AO_KNOB} + {FO_KNOB} + {TG_KNOB}: head -e x1 each, worker list x1 each, "
                f"validation x1 each + the FOLD-needs-ACC refusal)")


FI_KNOB = "GLM53_MLA_PREFILL_FUSED_INDEX"
ZP2_OLD = ZT2_NEW
ZP3_OLD = ZT3_NEW
ZP4_OLD = ZT4_NEW
ZP1_OLD = ZT1_NEW
ZP1_NEW = ZP1_OLD + """    # [mla-fused-index] the MLA prefill fused index pass (docs/OPT_DENSE.md), on both ranks. unset/empty/1 = on
    # (the deliberate default change of this kit: ONE Triton pass writes production's kv_indices bytes AND the
    # valid counts, -2.7 ms per 13,824-token forward_mqa; the whole kv_indices buffer == production's, byte for
    # byte); 0 = production's triton_convert + clamp + copy chain (the revert lever). Only read when
    # GLM53_MLA_PREFILL=1, by the site glm53_mla_prefill.py this kit ships (an older site module would silently
    # ignore =0 and keep the fused pass), so the value is refused here without that module.
    if [ -n "${GLM53_MLA_PREFILL_FUSED_INDEX:-}" ]; then
        case "$GLM53_MLA_PREFILL_FUSED_INDEX" in
            0|1) ;;
            *) echo "GLM53_MLA_PREFILL_FUSED_INDEX must be unset, 0 or 1 (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_MLA_PREFILL_FUSED_INDEX" "$TF_BUNDLE_DIR_HOST/site/glm53_mla_prefill.py" 2>/dev/null; then
            echo "GLM53_MLA_PREFILL_FUSED_INDEX requires the r16z4 site glm53_mla_prefill.py that reads it: $TF_BUNDLE_DIR_HOST/site/glm53_mla_prefill.py is an older one that would silently keep the fused pass" >&2
            return 2
        fi
    fi
"""
ZP2_NEW = ZP2_OLD + '        -e GLM53_MLA_PREFILL_FUSED_INDEX="${GLM53_MLA_PREFILL_FUSED_INDEX:-}" \\\n'
ZP3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16 "
           "GLM53_MOE_E4M3_LAYERS GLM53_MOE_E4M3_DOWN_LAYERS GLM53_MOE_E4M3_ACC GLM53_MOE_E4M3_FOLD_SHARED "
           "GLM53_MOE_E4M3_TOKGATHER GLM53_MLA_PREFILL_FUSED_INDEX; do  # [tf-exl3-fork]\n")
ZP4_NEW = ("# [mla-fused-index] GLM53_MLA_PREFILL_FUSED_INDEX (unset/1 = the fused index pass ON, the deliberate\n"
           "#   default change of this kit: one Triton pass writes production's kv_indices bytes AND the valid\n"
           "#   counts; 0 = production's triton_convert + clamp + copy chain, the revert lever) dials the MLA prefill\n"
           "#   (only read when GLM53_MLA_PREFILL=1; the whole kv_indices buffer is production's either way).\n"
           "#   docs/OPT_DENSE.md.\n") + ZP4_OLD
ZP_EDITS = (("ZP1 validation", ZP1_OLD, ZP1_NEW, 1), ("ZP2 head -e", ZP2_OLD, ZP2_NEW, 1),
            ("ZP3 worker list", ZP3_OLD, ZP3_NEW, 1), ("ZP4 note", ZP4_OLD, ZP4_NEW, 1))


def add_mla_fused_index(s: str) -> tuple[str, str]:
    """the moe-opt stage -> + ZP1-ZP4 (the MLA fused index switch), with flashkda's kind of checks."""
    if re.search(rf"\b{FI_KNOB}\b", s):
        sys.exit(f"ABORT: {FI_KNOB} already in the stage")
    for name, old, _new, n in ZP_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: mla-fused-index {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZP_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the mla-fused-index edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {FI_KNOB}="${{{FI_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(FI_KNOB) if m else 0
    val = s2.count('"GLM53_MLA_PREFILL_FUSED_INDEX must be unset, 0 or 1')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {FI_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZP_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the mla-fused-index stage differs from the incoming stage outside ZP1-ZP4")
    return s2, (f"+ mla-fused-index ZP1-ZP4 ({FI_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")


AG_KNOB = "GLM53_DENSE_W8A8_FP8AG"
ZG2_OLD = ZP2_NEW
ZG3_OLD = ZP3_NEW
ZG4_OLD = ZP4_NEW
ZG1_OLD = ZP1_NEW
ZG1_NEW = ZG1_OLD + """    # [w8a8-fp8ag] the fp8 sequence-parallel all-gather into a served KDA in_proj (docs/OPT_DENSE.md), on both
    # ranks. Only read when GLM53_DENSE_W8A8=1. unset/0 = the bf16 gather (production's bytes); 1 = the attention
    # gather of a KDA layer whose in_proj_qkvbfg_a is served carries per-token fp8 + scales (half the bytes; the
    # in_proj output is bitwise the W8A8 result). A PAIRED COLLECTIVE: each layer's path is the MIN over the TP
    # group of the local verdicts (one CPU vote per layer) - a rank whose W8A8 install was refused never joins the
    # vote and its peer BLOCKS at the first sequence-parallel forward, so boot_checks pair-gates the
    # "FP8 all-gather installed" line on BOTH ranks before every traffic check (and the refusal is logged at
    # ERROR). Anything else would stop both containers inside the bundle, so it is refused here - and the value
    # needs the r16z4 overlay fp8_w8a8.py that reads it (an older one would silently keep the bf16 gather).
    if [ -n "${GLM53_DENSE_W8A8_FP8AG:-}" ]; then
        case "$GLM53_DENSE_W8A8_FP8AG" in
            0|1) ;;
            *) echo "GLM53_DENSE_W8A8_FP8AG must be unset, 0 or 1 (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_DENSE_W8A8_FP8AG" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_FP8AG requires the r16z4 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently keep the bf16 gather" >&2
            return 2
        fi
    fi
"""
ZG2_NEW = ZG2_OLD + '        -e GLM53_DENSE_W8A8_FP8AG="${GLM53_DENSE_W8A8_FP8AG:-}" \\\n'
ZG3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16 "
           "GLM53_MOE_E4M3_LAYERS GLM53_MOE_E4M3_DOWN_LAYERS GLM53_MOE_E4M3_ACC GLM53_MOE_E4M3_FOLD_SHARED "
           "GLM53_MOE_E4M3_TOKGATHER GLM53_MLA_PREFILL_FUSED_INDEX GLM53_DENSE_W8A8_FP8AG; do  "
           "# [tf-exl3-fork]\n")
ZG4_NEW = ("# [w8a8-fp8ag] GLM53_DENSE_W8A8_FP8AG=1 (unset/0 = the bf16 gather, production's bytes; only read when\n"
           "#   GLM53_DENSE_W8A8=1) halves the sequence-parallel attention-gather bytes of a KDA layer whose\n"
           "#   in_proj_qkvbfg_a is served (per-token fp8 + scales instead of bf16; the in_proj output is bitwise\n"
           "#   the W8A8 result). PAIRED collective: boot_checks requires the \"FP8 all-gather installed\" line on\n"
           "#   BOTH ranks before traffic. docs/OPT_DENSE.md.\n") + ZG4_OLD
ZG_EDITS = (("ZG1 validation", ZG1_OLD, ZG1_NEW, 1), ("ZG2 head -e", ZG2_OLD, ZG2_NEW, 1),
            ("ZG3 worker list", ZG3_OLD, ZG3_NEW, 1), ("ZG4 note", ZG4_OLD, ZG4_NEW, 1))


def add_w8a8_fp8ag(s: str) -> tuple[str, str]:
    """the mla-fused-index stage -> + ZG1-ZG4 (the fp8 all-gather switch), with flashkda's kind of checks."""
    if re.search(rf"\b{AG_KNOB}\b", s):
        sys.exit(f"ABORT: {AG_KNOB} already in the stage")
    for name, old, _new, n in ZG_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: w8a8-fp8ag {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZG_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the w8a8-fp8ag edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {AG_KNOB}="${{{AG_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(AG_KNOB) if m else 0
    val = s2.count('"GLM53_DENSE_W8A8_FP8AG must be unset, 0 or 1')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {AG_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZG_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the w8a8-fp8ag stage differs from the incoming stage outside ZG1-ZG4")
    return s2, (f"+ w8a8-fp8ag ZG1-ZG4 ({AG_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")


HL_KNOB = "GLM53_DENSE_W8A8_HILO"
HS_KNOB = "GLM53_DENSE_W8A8_HILO_SEL"
ZH2_OLD = ZG2_NEW
ZH3_OLD = ZG3_NEW
ZH4_OLD = ZG4_NEW
ZH1_OLD = ZG1_NEW
ZH1_NEW = ZH1_OLD + """    # [w8a8-hilo] the hi+lo W8A8 knobs (docs/OPT_DENSE.md: the e4m3 residual of a few STABLE outlier input
    # channels is appended as extra K columns of the same GEMM), on both ranks. Only read when GLM53_DENSE_W8A8=1.
    # GLM53_DENSE_W8A8_HILO: unset = off (the W8A8 path as shipped); else a comma list of
    # "<group>.<proj>:<channels>" with channels a multiple of 16 in 16..2048 (the reviewed HA set:
    # kda.o_proj:256,shared.down_proj:128,dense.down_proj:512,mla.q_b_proj:256,mla.o_proj:512; draft.fc is NOT
    # accepted: NO-GO until its acceptance is measured). GLM53_DENSE_W8A8_HILO_SEL: unset/call = the channel set
    # is re-picked per call; first = frozen at each layer's first real served call (the A/B mode). Anything else
    # would stop both containers inside the bundle, so it is refused here - and both knobs need the r16z4 overlay
    # fp8_w8a8.py that reads them (an older one would silently serve the plain W8A8 path).
    if [ -n "${GLM53_DENSE_W8A8_HILO:-}" ]; then
        __zh_it="${GLM53_DENSE_W8A8_HILO},"
        while [ -n "$__zh_it" ]; do
            __zh_p="${__zh_it%%,*}"; __zh_it="${__zh_it#*,}"
            case "$__zh_p" in
                "") ;;
                kda.in_proj_qkvbfg_a:*|kda.o_proj:*|mla.fused_qkv_a_proj:*|mla.q_b_proj:*|mla.o_proj:*|shared.gate_up_proj:*|shared.down_proj:*|dense.gate_up_proj:*|dense.down_proj:*)
                    case "${__zh_p#*:}" in
                        ""|*[!0-9]*) echo "GLM53_DENSE_W8A8_HILO: channels must be a positive integer (value not printed)" >&2; return 2;;
                        *) __zh_c="${__zh_p#*:}"; [ "$__zh_c" -ge 16 ] && [ "$__zh_c" -le 2048 ] && [ $((__zh_c % 16)) -eq 0 ] || { echo "GLM53_DENSE_W8A8_HILO: channels must be a multiple of 16 in 16..2048 (value not printed)" >&2; return 2; };;
                    esac;;
                *) echo "GLM53_DENSE_W8A8_HILO: a name that is not one of the module's served projections (draft.fc is not accepted: NO-GO until its acceptance is measured) (value not printed)" >&2; return 2;;
            esac
        done
        if ! grep -qF "GLM53_DENSE_W8A8_HILO" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_HILO requires the r16z4 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently serve the plain W8A8 path" >&2
            return 2
        fi
    fi
    if [ -n "${GLM53_DENSE_W8A8_HILO_SEL:-}" ]; then
        case "$GLM53_DENSE_W8A8_HILO_SEL" in
            call|first) ;;
            *) echo "GLM53_DENSE_W8A8_HILO_SEL must be unset, call or first (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_DENSE_W8A8_HILO_SEL" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_HILO_SEL requires the r16z4 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently re-pick per call" >&2
            return 2
        fi
    fi
"""
ZH2_NEW = ZH2_OLD + '        -e GLM53_DENSE_W8A8_HILO="${GLM53_DENSE_W8A8_HILO:-}" \\\n' \
                    '        -e GLM53_DENSE_W8A8_HILO_SEL="${GLM53_DENSE_W8A8_HILO_SEL:-}" \\\n'
ZH3_NEW = ("             GLM53_DEC_SMALLOPS_KINDS GLM53_KPOOL_RING GLM53_KDA_STRIDED_QKV GLM53_KPOOL_DROP_LOWEST "
           "GLM53_KDA_FLASHKDA GLM53_MHC_SP GLM53_MOE_E4M3 GLM53_DENSE_W8A8 GLM53_KDA_FLASHKDA_V GLM53_MHC_SP2 "
           "GLM53_DENSE_W8A8_GEMM GLM53_DENSE_W8A8_ONLY GLM53_MOE_E4M3_DOWN GLM53_MOE_FUSED16 "
           "GLM53_MOE_E4M3_LAYERS GLM53_MOE_E4M3_DOWN_LAYERS GLM53_MOE_E4M3_ACC GLM53_MOE_E4M3_FOLD_SHARED "
           "GLM53_MOE_E4M3_TOKGATHER GLM53_MLA_PREFILL_FUSED_INDEX GLM53_DENSE_W8A8_FP8AG GLM53_DENSE_W8A8_HILO "
           "GLM53_DENSE_W8A8_HILO_SEL; do  # [tf-exl3-fork]\n")
ZH4_NEW = ("# [w8a8-hilo] GLM53_DENSE_W8A8_HILO (unset = off; else a comma list of \"<group>.<proj>:<channels>\",\n"
           "#   channels a multiple of 16 in 16..2048 - the e4m3 residual of those STABLE outlier input channels is\n"
           "#   appended as extra K columns of the same GEMM: the W8A8 quality lever, e.g. the reviewed HA set\n"
           "#   kda.o_proj:256,shared.down_proj:128,dense.down_proj:512,mla.q_b_proj:256,mla.o_proj:512) and\n"
           "#   GLM53_DENSE_W8A8_HILO_SEL (unset/call = per call; first = frozen at each layer's first real served\n"
           "#   call) dial the W8A8 prefill dense GEMMs (only read when GLM53_DENSE_W8A8=1). draft.fc is NOT\n"
           "#   accepted (NO-GO until its acceptance is measured). docs/OPT_DENSE.md.\n") + ZH4_OLD
ZH_EDITS = (("ZH1 validation", ZH1_OLD, ZH1_NEW, 1), ("ZH2 head -e", ZH2_OLD, ZH2_NEW, 1),
            ("ZH3 worker list", ZH3_OLD, ZH3_NEW, 1), ("ZH4 note", ZH4_OLD, ZH4_NEW, 1))


def add_w8a8_hilo(s: str) -> tuple[str, str]:
    """the w8a8-fp8ag stage -> + ZH1-ZH4 (the hi+lo W8A8 knobs), with flashkda's kind of checks."""
    for k in (HL_KNOB, HS_KNOB):
        if re.search(rf"\b{k}\b", s):
            sys.exit(f"ABORT: {k} already in the stage")
    for name, old, _new, n in ZH_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: w8a8-hilo {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZH_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the w8a8-hilo edits: {r.stderr.decode()}")
    for k in (HL_KNOB, HS_KNOB):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count('"GLM53_DENSE_W8A8_HILO_SEL must be unset, call or first')
    if val != 1:
        sys.exit(f"ABORT: {HS_KNOB} validation x{val} (need 1)")
    val = s2.count("GLM53_DENSE_W8A8_HILO: a name that is not one of the module's served projections")
    if val != 1:
        sys.exit(f"ABORT: {HL_KNOB} name validation x{val} (need 1)")
    back = s2
    for _name, old, new, n in ZH_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the w8a8-hilo stage differs from the incoming stage outside ZH1-ZH4")
    return s2, (f"+ w8a8-hilo ZH1-ZH4 ({HL_KNOB} + {HS_KNOB}: head -e x1 each, worker list x1 each, validation x1 each)")


# ---- r16z5 (opt-decodekit): the r16z4 stage + XL1-XL4, the exact-length FA2 planner switch GLM53_MLA_EXACT_LENS
# (docs/MLA_EXACT_LENS.md), anchored on the ZH results (the chain continued: ... -> ZL -> ZT -> ZP -> ZG -> ZH ->
# XL). Unset/empty/0 = production byte for byte (the bundle skips / stocks the overlay); forwarded to BOTH ranks.
# DEFAULT OFF: decode numerics change by design (a deliberate knob, not a default change).
XL_KNOB = "GLM53_MLA_EXACT_LENS"
XL1_OLD = ZH1_NEW
XL1_NEW = XL1_OLD + """    # [mla-exactlens] the exact-length FA2 sparse-MLA plan (docs/MLA_EXACT_LENS.md), on both ranks. Unset/empty/0 =
    # production's plan (index_topk + ctx % kpool keys per row once ctx >= index_topk: 4 more than the kpool indexer
    # selects, so every such decode row also attends slot-0 copies and the next row's / a stale step's keys); 1 = the
    # overlay patch_mla_exactlens.py (installs glm53_mla_exactlens.py and arms integrate.py), run by patch_tf_bundle.py.
    # Anything else would stop both containers inside the bundle, so it is refused here with its name.
    if [ -n "${GLM53_MLA_EXACT_LENS:-}" ]; then
        _glm53_validate_bool_flag GLM53_MLA_EXACT_LENS "$GLM53_MLA_EXACT_LENS" || return
    fi
    if [ "${GLM53_MLA_EXACT_LENS:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_mla_exactlens.py" ] \
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_mla_exactlens.py" ]; then
            echo "GLM53_MLA_EXACT_LENS=1 requires $TF_BUNDLE_DIR_HOST/overlay/{patch_mla_exactlens.py,glm53_mla_exactlens.py}" >&2
            return 2
        fi
        if ! grep -qF '("patch_mla_exactlens.py", "GLM53_MLA_EXACT_LENS")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MLA_EXACT_LENS=1 requires $TF_BUNDLE_PATCH_HOST to run patch_mla_exactlens.py (the r16z5 patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
"""
XL2_OLD = ZH2_NEW
XL2_NEW = XL2_OLD + '        -e GLM53_MLA_EXACT_LENS="${GLM53_MLA_EXACT_LENS:-}" \\\n'
XL3_OLD = ZH3_NEW
XL3_NEW = ZH3_NEW.replace(" GLM53_DENSE_W8A8_HILO_SEL; do  # [tf-exl3-fork]",
                          " GLM53_DENSE_W8A8_HILO_SEL GLM53_MLA_EXACT_LENS; do  # [tf-exl3-fork]")
XL4_OLD = ZH4_NEW
XL4_NEW = ("# [mla-exactlens] GLM53_MLA_EXACT_LENS (unset/empty/0 = production's FA2 sparse-MLA plan, 4 keys per row past\n"
           "#   index_topk more than the indexer selected; 1 = the exact counts, overlay patch_mla_exactlens.py; decode\n"
           "#   numerics change by design, prefill on the exact kernel does not). docs/MLA_EXACT_LENS.md.\n") + XL4_OLD
XL_EDITS = (("XL1 validation", XL1_OLD, XL1_NEW, 1), ("XL2 head -e", XL2_OLD, XL2_NEW, 1),
            ("XL3 worker list", XL3_OLD, XL3_NEW, 1), ("XL4 note", XL4_OLD, XL4_NEW, 1))


def add_exactlens(s: str) -> tuple[str, str]:
    """the r16z4 stage -> + XL1-XL4 (the exact-length FA2 planner switch), with flashkda's kind of checks."""
    if re.search(rf"\b{XL_KNOB}\b", s):
        sys.exit(f"ABORT: {XL_KNOB} already in the stage")
    for name, old, _new, n in XL_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: mla-exactlens {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in XL_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the mla-exactlens edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {XL_KNOB}="${{{XL_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(XL_KNOB) if m else 0
    val = s2.count('_glm53_validate_bool_flag GLM53_MLA_EXACT_LENS')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {XL_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in XL_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the mla-exactlens stage differs from the incoming stage outside XL1-XL4")
    return s2, (f"+ mla-exactlens XL1-XL4 ({XL_KNOB}: head -e x1, worker list x1, validation x1)")


# ---- r16z5 (kdamhc-rev / w8a8layers / moe2-rev): FOUR more edit groups on top of the r16z4 chain, in the order
# XL -> ZU -> ZM -> ZK -> ZW:
#   ZU1-ZU4 the MLA prefill kv_indices write-back limit GLM53_MLA_PREFILL_KV_ROWS (docs/KDAMHC.md; the module's
#   DEFAULT = only the rows an FA2 call can read - the third deliberate default change of this kit; 'all' = the r16
#   full write-back, the revert lever; free-form value 'all' or an integer row count, only read with
#   GLM53_MLA_PREFILL=1, by the site glm53_mla_prefill.py this kit ships);
#   ZM1-ZM4 the fused mHC post+prenorm GEMM GLM53_MHC_FUSED (docs/KDAMHC.md; DEFAULT OFF; 1 needs the overlay pair +
#   the AOT extension and a bundle script that runs the patch; the rank-consistent fixed-seed self-check of
#   opt-kdamhc-rev 969dc13) and its dials GLM53_MHC_FUSED_ROUND_A (0|1, default 0 = the decode-consistent rounding;
#   only read with MHC_FUSED=1) and GLM53_MHC_FUSED_CFG (a decimal, default 9; the kernel tile cfg, only read with
#   MHC_FUSED=1) - a ONE-RANK install must never happen (the mHC op feeds the same all-reduce on both ranks):
#   boot_checks pair-gates the armed line like mhcsp;
#   ZK1-ZK4 the W8A8 layer/projection exclusion GLM53_DENSE_W8A8_SKIP_LAYERS (docs/OPT_W8A8LAYERS.md; a comma list
#   of '<layer>[:<name>]' with name a served projection, an extra name or a group; applied after ONLY; a malformed
#   value REFUSES the W8A8 install = production's path; only read with GLM53_DENSE_W8A8=1, by overlay/fp8_w8a8.py);
#   ZW1-ZW4 the lean fused mainloop GLM53_MOE_E4M3_MAINLOOP (docs/OPT_MOE2.md; DEFAULT OFF = SASS-identical to
#   opt-moe-rev for the shipped 24 kernels; 1 = fused variants + 8192, -2.5 ms per 13,824-token layer call,
#   intermediates bitwise; only read when GLM53_MOE_E4M3=1, by overlay/glm53_moe_e4m3.py that reads it) -
#   all forwarded to BOTH ranks, default = production byte for byte.
KR_KNOB = "GLM53_MLA_PREFILL_KV_ROWS"
ZU2_OLD = XL2_NEW
ZU3_OLD = XL3_NEW
ZU4_OLD = XL4_NEW
ZU1_OLD = XL1_NEW
ZU1_NEW = ZU1_OLD + """    # [kdamhc-kvrows] the MLA prefill kv_indices write-back limit (docs/KDAMHC.md), on both ranks. Only read when
    # GLM53_MLA_PREFILL=1, by the site glm53_mla_prefill.py this kit ships. The module's DEFAULT (unset/empty) =
    # ONLY the rows an FA2 call can read beyond its own are written back (rows [0, max(min_tokens, 1024) + 1); the
    # rest of a 13,824-row write (113 MB clamp + 113 MB copy, ~1.9 ms per MLA layer) is never read - the third
    # deliberate default change of this kit; 'all' = the r16 full write-back (the revert lever) and an integer N =
    # an explicit row count. Any other value is refused here (an older site module would silently keep the full
    # write-back, i.e. ignore the value).
    if [ -n "${GLM53_MLA_PREFILL_KV_ROWS:-}" ]; then
        case "$GLM53_MLA_PREFILL_KV_ROWS" in
            all) ;;
            *[!0-9]*) echo "GLM53_MLA_PREFILL_KV_ROWS must be unset, 'all' or a non-negative integer row count (value not printed)" >&2; return 2;;
        esac
        if ! grep -qF "GLM53_MLA_PREFILL_KV_ROWS" "$TF_BUNDLE_DIR_HOST/site/glm53_mla_prefill.py" 2>/dev/null; then
            echo "GLM53_MLA_PREFILL_KV_ROWS requires the r16z5 site glm53_mla_prefill.py that reads it: $TF_BUNDLE_DIR_HOST/site/glm53_mla_prefill.py is an older one that would silently keep the full write-back" >&2
            return 2
        fi
    fi
"""
ZU2_NEW = ZU2_OLD + '        -e GLM53_MLA_PREFILL_KV_ROWS="${GLM53_MLA_PREFILL_KV_ROWS:-}" \\\n'
ZU3_NEW = ZU3_OLD.replace(" GLM53_MLA_EXACT_LENS; do  # [tf-exl3-fork]",
                                                       " GLM53_MLA_EXACT_LENS GLM53_MLA_PREFILL_KV_ROWS; do  # [tf-exl3-fork]")
ZU4_NEW = ("# [kdamhc-kvrows] GLM53_MLA_PREFILL_KV_ROWS (unset = ONLY the rows an FA2 call can read - the deliberate\n"
           "#   default change of this kit; 'all' = the r16 full write-back, the revert lever; N = an explicit row\n"
           "#   count) limits the MLA prefill's kv_indices write-back (only read when GLM53_MLA_PREFILL=1). The rows\n"
           "#   kept are byte-identical to production's; decode never reads the rest. docs/KDAMHC.md.\n") + ZU4_OLD
ZU_EDITS = (("ZU1 validation", ZU1_OLD, ZU1_NEW, 1), ("ZU2 head -e", ZU2_OLD, ZU2_NEW, 1),
            ("ZU3 worker list", ZU3_OLD, ZU3_NEW, 1), ("ZU4 note", ZU4_OLD, ZU4_NEW, 1))


def add_kdamhc_kvrows(s: str) -> tuple[str, str]:
    """the exactlens stage -> + ZU1-ZU4 (the kv_rows knob), with flashkda's kind of checks."""
    if re.search(rf"\b{KR_KNOB}\b", s):
        sys.exit(f"ABORT: {KR_KNOB} already in the stage")
    for name, old, _new, n in ZU_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kdamhc-kvrows {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZU_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kdamhc-kvrows edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {KR_KNOB}="${{{KR_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(KR_KNOB) if m else 0
    val = s2.count('GLM53_MLA_PREFILL_KV_ROWS must be unset')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {KR_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZU_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kdamhc-kvrows stage differs from the incoming stage outside ZU1-ZU4")
    return s2, (f"+ kdamhc-kvrows ZU1-ZU4 ({KR_KNOB}: head -e x1, worker list x1, validation x1)")


MF_KNOB = "GLM53_MHC_FUSED"
MR_KNOB = "GLM53_MHC_FUSED_ROUND_A"
MC_KNOB = "GLM53_MHC_FUSED_CFG"
ZM2_OLD = ZU2_NEW
ZM3_OLD = ZU3_NEW
ZM4_OLD = ZU4_NEW
ZM1_OLD = ZU1_NEW
ZM1_NEW = ZM1_OLD + """    # [kdamhc-mhcfused] the fused mHC post + prenorm GEMM (docs/KDAMHC.md), on both ranks. Unset/empty/0 = stock
    # (the bundle does not run the overlay: nothing of the feature reaches site-packages); 1 = the bundle overlay
    # patch_mhc_fused.py (copies glm53_mhc_fused.py + its AOT extension into site-packages and arms integrate.py
    # AFTER glm53_moe_e4m3's block): the mHC post + prenorm-GEMM prefill kernel (residual_cur bitwise production's;
    # the 24 mixing logits in the decode branch's fp32 arithmetic; the opt-kdamhc-rev rank-consistent fixed-seed
    # self-check). A PAIRED FEATURE: a rank whose install was refused (or whose self-check uninstalled) must never
    # serve alone - boot_checks pair-gates the armed line on BOTH ranks before traffic. Anything else would stop
    # both containers inside the bundle, so it is refused here.
    if [ -n "${GLM53_MHC_FUSED:-}" ]; then
        _glm53_validate_bool_flag GLM53_MHC_FUSED "$GLM53_MHC_FUSED" || return
    fi
    if [ "${GLM53_MHC_FUSED:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_mhc_fused.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_mhc_fused.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so" ]; then
            echo "GLM53_MHC_FUSED=1 requires $TF_BUNDLE_DIR_HOST/overlay/{patch_mhc_fused.py,glm53_mhc_fused.py,glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so}" >&2
            return 2
        fi
        if ! grep -qF '("patch_mhc_fused.py", "GLM53_MHC_FUSED")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_MHC_FUSED=1 requires $TF_BUNDLE_PATCH_HOST to run patch_mhc_fused.py (the r16z5 patch_tf_bundle.py)" >&2
            return 2
        fi
        if [ -n "${GLM53_MHC_FUSED_ROUND_A:-}" ]; then
            case "$GLM53_MHC_FUSED_ROUND_A" in
                0|1) ;;
                *) echo "GLM53_MHC_FUSED_ROUND_A must be unset, 0 or 1 (value not printed)" >&2; return 2;;
            esac
            if ! grep -qF "GLM53_MHC_FUSED_ROUND_A" "$TF_BUNDLE_DIR_HOST/overlay/glm53_mhc_fused.py" 2>/dev/null; then
                echo "GLM53_MHC_FUSED_ROUND_A requires the r16z5 overlay glm53_mhc_fused.py that reads it" >&2
                return 2
            fi
        fi
        if [ -n "${GLM53_MHC_FUSED_CFG:-}" ]; then
            case "$GLM53_MHC_FUSED_CFG" in
                ''|*[!0-9]*) echo "GLM53_MHC_FUSED_CFG must be unset or a decimal kernel cfg (value not printed)" >&2; return 2;;
            esac
            if ! grep -qF "GLM53_MHC_FUSED_CFG" "$TF_BUNDLE_DIR_HOST/overlay/glm53_mhc_fused.py" 2>/dev/null; then
                echo "GLM53_MHC_FUSED_CFG requires the r16z5 overlay glm53_mhc_fused.py that reads it" >&2
                return 2
            fi
        fi
    fi
"""
ZM2_NEW = ZM2_OLD + '        -e GLM53_MHC_FUSED="${GLM53_MHC_FUSED:-}" \\\n' \
                    '        -e GLM53_MHC_FUSED_ROUND_A="${GLM53_MHC_FUSED_ROUND_A:-}" \\\n' \
                    '        -e GLM53_MHC_FUSED_CFG="${GLM53_MHC_FUSED_CFG:-}" \\\n'
ZM3_NEW = ZM3_OLD.replace(" GLM53_MLA_PREFILL_KV_ROWS; do  # [tf-exl3-fork]",
                          " GLM53_MLA_PREFILL_KV_ROWS GLM53_MHC_FUSED GLM53_MHC_FUSED_ROUND_A GLM53_MHC_FUSED_CFG; do  "
                          "# [tf-exl3-fork]")
ZM4_NEW = ("# [kdamhc-mhcfused] GLM53_MHC_FUSED=1 (unset/0 = stock) -> the bundle overlay patch_mhc_fused.py: the mHC\n"
           "#   post + prenorm-GEMM prefill kernel (residual_cur bitwise production's; the 24 mixing logits in the\n"
           "#   decode branch's fp32 arithmetic; rank-consistent fixed-seed self-check at load; PAIRED - boot_checks\n"
           "#   requires the armed line on BOTH ranks before traffic). GLM53_MHC_FUSED_ROUND_A (0|1, default 0 = the\n"
           "#   decode-consistent rounding) and GLM53_MHC_FUSED_CFG (decimal, default 9 = BM16/HT64 + register\n"
           "#   prefetch) dial the kernel (only read when GLM53_MHC_FUSED=1). docs/KDAMHC.md.\n") + ZM4_OLD
ZM_EDITS = (("ZM1 validation", ZM1_OLD, ZM1_NEW, 1), ("ZM2 head -e", ZM2_OLD, ZM2_NEW, 1),
            ("ZM3 worker list", ZM3_OLD, ZM3_NEW, 1), ("ZM4 note", ZM4_OLD, ZM4_NEW, 1))


def add_mhc_fused(s: str) -> tuple[str, str]:
    """the kv_rows stage -> + ZM1-ZM4 (the fused mHC switch + dials), with flashkda's kind of checks."""
    for k in (MF_KNOB, MR_KNOB, MC_KNOB):
        if re.search(rf"\b{k}\b", s):
            sys.exit(f"ABORT: {k} already in the stage")
    for name, old, _new, n in ZM_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kdamhc-mhcfused {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZM_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kdamhc-mhcfused edits: {r.stderr.decode()}")
    for k in (MF_KNOB, MR_KNOB, MC_KNOB):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count('"GLM53_MHC_FUSED_ROUND_A must be unset, 0 or 1')
    val2 = s2.count('"GLM53_MHC_FUSED_CFG must be unset or a decimal kernel cfg')
    if val != 1 or val2 != 1:
        sys.exit(f"ABORT: the mhc-fused dial validations x{val}/x{val2} (need 1/1)")
    back = s2
    for _name, old, new, n in ZM_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kdamhc-mhcfused stage differs from the incoming stage outside ZM1-ZM4")
    return s2, (f"+ kdamhc-mhcfused ZM1-ZM4 ({MF_KNOB} + {MR_KNOB} + {MC_KNOB}: head -e x1 each, worker list x1 each, "
                f"validation x1 each)")


SK_KNOB = "GLM53_DENSE_W8A8_SKIP_LAYERS"
ZK2_OLD = ZM2_NEW
ZK3_OLD = ZM3_NEW
ZK4_OLD = ZM4_NEW
ZK1_OLD = ZM1_NEW
ZK1_NEW = ZK1_OLD + """    # [w8a8layers] the W8A8 layer/projection exclusion (docs/OPT_W8A8LAYERS.md), on both ranks. Only read when
    # GLM53_DENSE_W8A8=1, by the r16z5 overlay fp8_w8a8.py that reads it (an older one would silently serve every
    # layer). unset = no layer excluded (production's path byte for byte); else a comma list of '<layer>[:<name>]'
    # with <layer> N or A-B (0..4095) and <name> a served "<group>.<projection>", an extra name (kda.f_b_proj,
    # kda.g_b_proj) or a group (kda|mla|dense|shared); applied AFTER GLM53_DENSE_W8A8_ONLY. A MALFORMED VALUE
    # REFUSES THE W8A8 INSTALL (production's dense FP8 path serves; the operator's typo must not half-serve).
    if [ -n "${GLM53_DENSE_W8A8_SKIP_LAYERS:-}" ]; then
        __zk_it="${GLM53_DENSE_W8A8_SKIP_LAYERS},"
        while [ -n "$__zk_it" ]; do
            __zk_p="${__zk_it%%,*}"; __zk_it="${__zk_it#*,}"
            __zk_rng="${__zk_p%%:*}"
            case "$__zk_rng" in
                ""|0) ;;
                *[!0-9-]*) echo "GLM53_DENSE_W8A8_SKIP_LAYERS: the layer part must be N or A-B (value not printed)" >&2; return 2;;
                *-*) case "${__zk_rng%%-*}${__zk_rng#*-}" in *[!0-9]*) echo "GLM53_DENSE_W8A8_SKIP_LAYERS: a malformed range (value not printed)" >&2; return 2;; esac
                     [ "${__zk_rng%%-*}" -le "${__zk_rng#*-}" ] || { echo "GLM53_DENSE_W8A8_SKIP_LAYERS: a descending range (value not printed)" >&2; return 2; }
                     [ "${__zk_rng#*-}" -le 4095 ] || { echo "GLM53_DENSE_W8A8_SKIP_LAYERS: a layer above 4095 (value not printed)" >&2; return 2; };;
                *) [ "$__zk_rng" -le 4095 ] || { echo "GLM53_DENSE_W8A8_SKIP_LAYERS: a layer above 4095 (value not printed)" >&2; return 2; };;
            esac
        done
        if ! grep -qF "GLM53_DENSE_W8A8_SKIP_LAYERS" "$TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py" 2>/dev/null; then
            echo "GLM53_DENSE_W8A8_SKIP_LAYERS requires the r16z5 overlay fp8_w8a8.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/fp8_w8a8.py is an older one that would silently serve every layer" >&2
            return 2
        fi
    fi
"""
ZK2_NEW = ZK2_OLD + '        -e GLM53_DENSE_W8A8_SKIP_LAYERS="${GLM53_DENSE_W8A8_SKIP_LAYERS:-}" \\\n'
ZK3_NEW = ZK3_OLD.replace(" GLM53_MHC_FUSED_CFG; do  # [tf-exl3-fork]",
                          " GLM53_MHC_FUSED_CFG GLM53_DENSE_W8A8_SKIP_LAYERS; do  # [tf-exl3-fork]")
ZK4_NEW = ("# [w8a8layers] GLM53_DENSE_W8A8_SKIP_LAYERS (unset = no layer excluded, production's path byte for byte;\n"
           "#   else a comma list of '<layer>[:<name>]' - the W8A8 path does not serve those layers/projections, they\n"
           "#   stay on production's Marlin path; applied after GLM53_DENSE_W8A8_ONLY; a malformed value REFUSES the\n"
           "#   W8A8 install) excludes W8A8 layers (only read when GLM53_DENSE_W8A8=1; the install line names the\n"
           "#   filter and the load summary counts skip_layers_at_load). docs/OPT_W8A8LAYERS.md.\n") + ZK4_OLD
ZK_EDITS = (("ZK1 validation", ZK1_OLD, ZK1_NEW, 1), ("ZK2 head -e", ZK2_OLD, ZK2_NEW, 1),
            ("ZK3 worker list", ZK3_OLD, ZK3_NEW, 1), ("ZK4 note", ZK4_OLD, ZK4_NEW, 1))


def add_w8a8_skip(s: str) -> tuple[str, str]:
    """the mhc-fused stage -> + ZK1-ZK4 (the W8A8 layer exclusion), with flashkda's kind of checks."""
    if re.search(rf"\b{SK_KNOB}\b", s):
        sys.exit(f"ABORT: {SK_KNOB} already in the stage")
    for name, old, _new, n in ZK_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: w8a8layers {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZK_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the w8a8layers edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {SK_KNOB}="${{{SK_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(SK_KNOB) if m else 0
    val = s2.count('GLM53_DENSE_W8A8_SKIP_LAYERS: the layer part must be N or A-B')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {SK_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZK_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the w8a8layers stage differs from the incoming stage outside ZK1-ZK4")
    return s2, (f"+ w8a8layers ZK1-ZK4 ({SK_KNOB}: head -e x1, worker list x1, validation x1)")


MS_KNOB = "GLM53_MOE_E4M3_MAINLOOP"
ZN2_OLD = ZK2_NEW
ZN3_OLD = ZK3_NEW
ZN4_OLD = ZK4_NEW
ZN1_OLD = ZK1_NEW
ZN1_NEW = ZN1_OLD + """    # [moe2] the lean fused mainloop of the e4m3 routed-MoE prefill (docs/OPT_MOE2.md), on both ranks. Only read
    # when GLM53_MOE_E4M3=1. unset/empty/0 = the shipped fused kernel (SASS-identical to opt-moe-rev for the 24
    # shipped kernels, byte for byte); 1 = fused variants + 8192: the same mainloop with fewer instructions per
    # stage (the lean trellis decode + a per-thread copy-address table; the same smem layout, fragments and mma
    # sequence, intermediates bitwise identical; -2.5 ms per 13,824-token layer call). Anything else would stop
    # both containers inside the bundle, so it is refused here - and the value needs the r16z5 overlay
    # glm53_moe_e4m3.py that reads it (an older one would silently serve the shipped mainloop).
    if [ -n "${GLM53_MOE_E4M3_MAINLOOP:-}" ]; then
        _glm53_validate_bool_flag GLM53_MOE_E4M3_MAINLOOP "$GLM53_MOE_E4M3_MAINLOOP" || return
        if ! grep -qF "GLM53_MOE_E4M3_MAINLOOP" "$TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py" 2>/dev/null; then
            echo "GLM53_MOE_E4M3_MAINLOOP requires the r16z5 overlay glm53_moe_e4m3.py that reads it: $TF_BUNDLE_DIR_HOST/overlay/glm53_moe_e4m3.py is an older one that would silently serve the shipped mainloop" >&2
            return 2
        fi
    fi
"""
ZN2_NEW = ZN2_OLD + '        -e GLM53_MOE_E4M3_MAINLOOP="${GLM53_MOE_E4M3_MAINLOOP:-}" \\\n'
ZN3_NEW = ZN3_OLD.replace(" GLM53_DENSE_W8A8_SKIP_LAYERS; do  # [tf-exl3-fork]",
                          " GLM53_DENSE_W8A8_SKIP_LAYERS GLM53_MOE_E4M3_MAINLOOP; do  # [tf-exl3-fork]")
ZN4_NEW = ("# [moe2] GLM53_MOE_E4M3_MAINLOOP=1 (unset/0 = the shipped fused kernel, SASS-identical to opt-moe-rev for\n"
           "#   the shipped 24 kernels; 1 = the lean fused mainloop: fused variants + 8192, -2.5 ms per 13,824-token\n"
           "#   layer call, intermediates bitwise identical) dials the e4m3 routed-MoE prefill (only read when\n"
           "#   GLM53_MOE_E4M3=1). Static smem 2,064 B for the TG variants (100,880 of 101,376 B per block - 496 B\n"
           "#   headroom). docs/OPT_MOE2.md.\n") + ZN4_OLD
ZN_EDITS = (("ZN1 validation", ZN1_OLD, ZN1_NEW, 1), ("ZN2 head -e", ZN2_OLD, ZN2_NEW, 1),
            ("ZN3 worker list", ZN3_OLD, ZN3_NEW, 1), ("ZN4 note", ZN4_OLD, ZN4_NEW, 1))


def add_moe_mainloop(s: str) -> tuple[str, str]:
    """the w8a8layers stage -> + ZW1-ZW4 (the lean mainloop switch), with flashkda's kind of checks."""
    if re.search(rf"\b{MS_KNOB}\b", s):
        sys.exit(f"ABORT: {MS_KNOB} already in the stage")
    for name, old, _new, n in ZN_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: moe2-rev {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in ZN_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the moe2-rev edits: {r.stderr.decode()}")
    head = s2.count(f'        -e {MS_KNOB}="${{{MS_KNOB}:-}}" \\\n')
    m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
    worker = m.group(1).replace("\\\n", " ").split().count(MS_KNOB) if m else 0
    val = s2.count('_glm53_validate_bool_flag GLM53_MOE_E4M3_MAINLOOP')
    if (head, worker, val) != (1, 1, 1):
        sys.exit(f"ABORT: {MS_KNOB} placement head={head} worker={worker} validation={val} (need 1, 1, 1)")
    back = s2
    for _name, old, new, n in ZN_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the moe2-rev stage differs from the incoming stage outside ZN1-ZN4")
    return s2, (f"+ moe2-rev ZN1-ZN4 ({MS_KNOB}: head -e x{head}, worker list x{worker}, validation x{val})")


# ---- r16z6 (opt-decode): the decode KDA_LAZY knobs (docs/DEC_KDA_LAZY.md) and the verify-trimming collector
# (docs/OPT_DECODE.md 4) on top of the r16z5 stage, in the chain's order (... -> ZN -> KL -> VT). The six knobs
# default to UNSET = production byte for byte. They are read by two modules the bundle SHIPS IN site/ (setup.py
# py_modules glm53_kda_lazy.py + glm53_vtrim_stats.py, imported by the site integrate.py in every process, inert
# when unset): NOT an overlay install, so the checks below require the bundle site to carry the module instead of
# an overlay file.
KL_KNOB = "GLM53_DEC_KDA_LAZY"
KL2_OLD = ZN2_NEW
KL3_OLD = ZN3_NEW
KL4_OLD = ZN4_NEW
KL1_OLD = ZN1_NEW
KL1_NEW = KL1_OLD + """    # [kdalazy] one KDA recurrent-state store per verify step instead of one per row (docs/DEC_KDA_LAZY.md), on
    # both ranks. Unset/empty/0 = stock (the verify keeps production's per-row state stores); 1 = the site module
    # glm53_kda_lazy.py arms itself (an in-process self-check with a repair fallback: the first 64 commits, then
    # one in _VERIFY_EVERY). GLM53_DEC_KDA_LAZY_VERIFY / _VERIFY_EVERY dial that self-check: unset/empty = the
    # module defaults (64 / 1024), else a non-negative integer (0 = never; repair mode is exact but ~5 ms/step
    # SLOWER than production, so a self-check difference is a rollback trigger, not a harmless fallback). Anything
    # else would stop both containers inside the bundle, so it is refused here - and =1 needs the bundle site/ to
    # carry the module (an older bundle without it would silently run production's stores).
    if [ -n "${GLM53_DEC_KDA_LAZY:-}" ]; then
        _glm53_validate_bool_flag GLM53_DEC_KDA_LAZY "$GLM53_DEC_KDA_LAZY" || return
    fi
    if [ "${GLM53_DEC_KDA_LAZY:-}" = "1" ] && [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_kda_lazy.py" ]; then
        echo "GLM53_DEC_KDA_LAZY=1 requires $TF_BUNDLE_DIR_HOST/site/glm53_kda_lazy.py (the bundle site/ this kit ships)" >&2
        return 2
    fi
    for __kl_n in GLM53_DEC_KDA_LAZY_VERIFY GLM53_DEC_KDA_LAZY_VERIFY_EVERY; do
        if [ -n "$(eval "echo \\${$__kl_n:-}")" ]; then
            case "$(eval "echo \\${$__kl_n:-}")" in
                *[!0-9]*) echo "$__kl_n must be unset/empty or a non-negative integer (value not printed)" >&2; return 2;;
            esac
        fi
    done
"""
KL2_NEW = KL2_OLD + ('        -e GLM53_DEC_KDA_LAZY="${GLM53_DEC_KDA_LAZY:-}" \\\n'
                     '        -e GLM53_DEC_KDA_LAZY_VERIFY="${GLM53_DEC_KDA_LAZY_VERIFY:-}" \\\n'
                     '        -e GLM53_DEC_KDA_LAZY_VERIFY_EVERY="${GLM53_DEC_KDA_LAZY_VERIFY_EVERY:-}" \\\n')
KL3_NEW = KL3_OLD.replace(" GLM53_MOE_E4M3_MAINLOOP; do  # [tf-exl3-fork]",
                          " GLM53_MOE_E4M3_MAINLOOP GLM53_DEC_KDA_LAZY GLM53_DEC_KDA_LAZY_VERIFY"
                          " GLM53_DEC_KDA_LAZY_VERIFY_EVERY; do  # [tf-exl3-fork]")
KL4_NEW = ("# [kdalazy] GLM53_DEC_KDA_LAZY=1 (unset/empty/0 = stock) -> the site module glm53_kda_lazy.py: the KDA\n"
           "#   verify stores every row's raw inputs and one eager commit after sampling recomputes the accepted\n"
           "#   prefix with production's arithmetic (bitwise; the cache then holds production's bytes). The built-in\n"
           "#   self-check (GLM53_DEC_KDA_LAZY_VERIFY first commits, default 64, then one in _VERIFY_EVERY, default\n"
           "#   1024) compares fast vs production's kernel byte for byte; a DIFFERENCE puts the commit in repair mode\n"
           "#   (exact, ~5 ms/step SLOWER: a rollback trigger, boot_checks treats the 'differ' line as a refusal).\n"
           "#   On BOTH ranks. docs/DEC_KDA_LAZY.md.\n") + KL4_OLD
KL_EDITS = (("KL1 validation", KL1_OLD, KL1_NEW, 1), ("KL2 head -e", KL2_OLD, KL2_NEW, 1),
            ("KL3 worker list", KL3_OLD, KL3_NEW, 1), ("KL4 note", KL4_OLD, KL4_NEW, 1))


def add_kda_lazy(s: str) -> tuple[str, str]:
    """the moe2-rev (r16z5) stage -> + KL1-KL4 (the decode KDA_LAZY knobs), with flashkda's kind of checks."""
    if re.search(rf"\b{KL_KNOB}\b", s):
        sys.exit(f"ABORT: {KL_KNOB} already in the stage")
    for name, old, _new, n in KL_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kdalazy {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in KL_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kdalazy edits: {r.stderr.decode()}")
    for k in (KL_KNOB, "GLM53_DEC_KDA_LAZY_VERIFY", "GLM53_DEC_KDA_LAZY_VERIFY_EVERY"):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count('_glm53_validate_bool_flag GLM53_DEC_KDA_LAZY "$GLM53_DEC_KDA_LAZY"')
    sit = s2.count("site/glm53_kda_lazy.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {KL_KNOB} validation x{val} (need 1), site check x{sit} (need 2)")
    back = s2
    for _name, old, new, n in KL_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kdalazy stage differs from the incoming stage outside KL1-KL4")
    return s2, (f"+ kdalazy KL1-KL4 ({KL_KNOB} + _VERIFY + _VERIFY_EVERY: head -e x1 each, worker list x1 each, "
                f"validation x1)")


VT_KNOB = "GLM53_DEC_VTRIM_STATS"
VT2_OLD = KL2_NEW
VT3_OLD = KL3_NEW
VT4_OLD = KL4_NEW
VT1_OLD = KL1_NEW
VT1_NEW = VT1_OLD + """    # [vtrim] the log-only collector for per-step verify trimming (docs/OPT_DECODE.md 4), on both ranks.
    # Unset/empty/0 = off (sampling and outputs untouched; nothing recorded). A positive integer N = the site
    # module glm53_vtrim_stats.py records 16 probabilities/counts per request and verify step (no token ids) into
    # a GPU ring written every N verify calls to GLM53_DEC_VTRIM_STATS_FILE (default the bind-mounted vLLM cache
    # dir, ~0.4 ms/step while on). _CAP (a positive integer, default 200000) sizes the ring; _FILE is a path.
    # Anything else would stop both containers inside the bundle, so it is refused here - and a value needs the
    # bundle site/ to carry the module.
    if [ -n "${GLM53_DEC_VTRIM_STATS:-}" ]; then
        case "$GLM53_DEC_VTRIM_STATS" in
            0) ;;
            *[!0-9]*) echo "GLM53_DEC_VTRIM_STATS must be unset/empty/0 or a positive integer (value not printed)" >&2; return 2;;
        esac
    fi
    for __vt_n in GLM53_DEC_VTRIM_STATS_CAP GLM53_DEC_VTRIM_STATS_FILE; do
        if [ -n "$(eval "echo \\${$__vt_n:-}")" ]; then
            case "$__vt_n" in
                GLM53_DEC_VTRIM_STATS_CAP) case "$(eval "echo \\${$__vt_n:-}")" in
                    0|*[!0-9]*) echo "GLM53_DEC_VTRIM_STATS_CAP must be a positive integer (value not printed)" >&2; return 2;;
                esac;;
                *) case "$(eval "echo \\${$__vt_n:-}")" in
                    -*|*/) echo "GLM53_DEC_VTRIM_STATS_FILE must be a path (value not printed)" >&2; return 2;;
                esac;;
            esac
        fi
    done
    if [ -n "${GLM53_DEC_VTRIM_STATS:-}" ] && [ "${GLM53_DEC_VTRIM_STATS:-}" != "0" ] \\
       && [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_vtrim_stats.py" ]; then
        echo "GLM53_DEC_VTRIM_STATS requires $TF_BUNDLE_DIR_HOST/site/glm53_vtrim_stats.py (the bundle site/ this kit ships)" >&2
        return 2
    fi
"""
VT2_NEW = VT2_OLD + ('        -e GLM53_DEC_VTRIM_STATS="${GLM53_DEC_VTRIM_STATS:-}" \\\n'
                     '        -e GLM53_DEC_VTRIM_STATS_CAP="${GLM53_DEC_VTRIM_STATS_CAP:-}" \\\n'
                     '        -e GLM53_DEC_VTRIM_STATS_FILE="${GLM53_DEC_VTRIM_STATS_FILE:-}" \\\n')
VT3_NEW = VT3_OLD.replace(" GLM53_DEC_KDA_LAZY_VERIFY_EVERY; do  # [tf-exl3-fork]",
                          " GLM53_DEC_KDA_LAZY_VERIFY_EVERY GLM53_DEC_VTRIM_STATS GLM53_DEC_VTRIM_STATS_CAP"
                          " GLM53_DEC_VTRIM_STATS_FILE; do  # [tf-exl3-fork]")
VT4_NEW = ("# [vtrim] GLM53_DEC_VTRIM_STATS=N (unset/empty/0 = off) -> the site module glm53_vtrim_stats.py: a\n"
           "#   LOG-ONLY collector (outputs untouched, ~0.4 ms/step while on) recording 16 probabilities/counts per\n"
           "#   request and verify step (no token ids) into a GPU ring written every N verify calls to\n"
           "#   GLM53_DEC_VTRIM_STATS_FILE (default the bind-mounted vLLM cache dir) - the calibration data for the\n"
           "#   per-step verify-trimming rule. _CAP sizes the ring, _FILE overrides the path. On BOTH ranks.\n"
           "#   docs/OPT_DECODE.md section 4.\n") + VT4_OLD
VT_EDITS = (("VT1 validation", VT1_OLD, VT1_NEW, 1), ("VT2 head -e", VT2_OLD, VT2_NEW, 1),
            ("VT3 worker list", VT3_OLD, VT3_NEW, 1), ("VT4 note", VT4_OLD, VT4_NEW, 1))


def add_vtrim(s: str) -> tuple[str, str]:
    """the kdalazy stage -> + VT1-VT4 (the verify-trimming collector knobs), with flashkda's kind of checks."""
    if re.search(rf"\b{VT_KNOB}\b", s):
        sys.exit(f"ABORT: {VT_KNOB} already in the stage")
    for name, old, _new, n in VT_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: vtrim {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in VT_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the vtrim edits: {r.stderr.decode()}")
    for k in (VT_KNOB, "GLM53_DEC_VTRIM_STATS_CAP", "GLM53_DEC_VTRIM_STATS_FILE"):
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_DEC_VTRIM_STATS must be unset/empty/0 or a positive integer")
    sit = s2.count("site/glm53_vtrim_stats.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {VT_KNOB} validation x{val} (need 1), site check x{sit} (need 1)")
    back = s2
    for _name, old, new, n in VT_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the vtrim stage differs from the incoming stage outside VT1-VT4")
    return s2, (f"+ vtrim VT1-VT4 ({VT_KNOB} + _CAP + _FILE: head -e x1 each, worker list x1 each, validation x1)")


# ---- r16z6sv (decode4 track B): GLM53_SPEC_VTRIM (+ _TAU / _MIN / _LOG), per-step verify trimming from the DFlash2
# drafter's confidence (docs/SPEC_VTRIM.md), on top of the r16z6 stage (... -> KL -> VT -> SV). The four knobs default
# to UNSET = production byte for byte; they are read by the site module glm53_spec_vtrim.py (setup.py py_modules,
# imported by the site integrate.py in every process, inert when unset).
SV_KNOB = "GLM53_SPEC_VTRIM"
SV_NAMES = ("GLM53_SPEC_VTRIM", "GLM53_SPEC_VTRIM_TAU", "GLM53_SPEC_VTRIM_MIN", "GLM53_SPEC_VTRIM_LOG")
SV2_OLD = VT2_NEW
SV3_OLD = VT3_NEW
SV4_OLD = VT4_NEW
SV1_OLD = VT1_NEW
SV1_NEW = SV1_OLD + """    # [specvtrim] per-step verify trimming from the DFlash2 drafter's own confidence (docs/SPEC_VTRIM.md), on both
    # ranks. Unset/empty/off/0 = production; shadow = log what trimming would do (outputs untouched); on/1 = trim
    # (drafts after the first position whose survival estimate is < TAU are not verified: exact, the rejection
    # sampler sees placeholders and the dead rows' routed experts are skipped). _TAU a decimal in [0, 1] (default
    # 0.25), _MIN 0..7 (default 0), _LOG a non-negative integer (stats line period, default 2000). Anything else
    # would stop both containers inside the bundle (mode on refuses to start rather than run on one rank only),
    # so it is refused here - and a mode needs the bundle site/ to carry the module.
    if [ -n "${GLM53_SPEC_VTRIM:-}" ]; then
        case "$GLM53_SPEC_VTRIM" in
            off|0|shadow|on|1) ;;
            *) echo "GLM53_SPEC_VTRIM must be unset/empty/off/0, shadow or on/1 (value not printed)" >&2; return 2;;
        esac
    fi
    if [ -n "${GLM53_SPEC_VTRIM_TAU:-}" ] && ! [[ "$GLM53_SPEC_VTRIM_TAU" =~ ^(0(\\.[0-9]+)?|1(\\.0+)?|\\.[0-9]+)$ ]]; then
        echo "GLM53_SPEC_VTRIM_TAU must be a decimal in [0, 1] (value not printed)" >&2; return 2
    fi
    if [ -n "${GLM53_SPEC_VTRIM_MIN:-}" ] && ! [[ "$GLM53_SPEC_VTRIM_MIN" =~ ^[0-7]$ ]]; then
        echo "GLM53_SPEC_VTRIM_MIN must be 0..7 (value not printed)" >&2; return 2
    fi
    if [ -n "${GLM53_SPEC_VTRIM_LOG:-}" ] && ! [[ "$GLM53_SPEC_VTRIM_LOG" =~ ^[0-9]+$ ]]; then
        echo "GLM53_SPEC_VTRIM_LOG must be a non-negative integer (value not printed)" >&2; return 2
    fi
    case "${GLM53_SPEC_VTRIM:-}" in
        shadow|on|1) if [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_spec_vtrim.py" ]; then
            echo "GLM53_SPEC_VTRIM requires $TF_BUNDLE_DIR_HOST/site/glm53_spec_vtrim.py (the bundle site/ this kit ships)" >&2
            return 2
        fi;;
    esac
"""
SV2_NEW = SV2_OLD + "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in SV_NAMES)
SV3_NEW = SV3_OLD.replace(" GLM53_DEC_VTRIM_STATS_FILE; do  # [tf-exl3-fork]",
                          " GLM53_DEC_VTRIM_STATS_FILE " + " ".join(SV_NAMES) + "; do  # [tf-exl3-fork]")
SV4_NEW = ("# [specvtrim] GLM53_SPEC_VTRIM=shadow|on (unset/empty/off/0 = production) -> the site module\n"
           "#   glm53_spec_vtrim.py: n* per request from the DFlash2 selector's distributions (S_i = qd_1..qd_{i-1} x\n"
           "#   qmax_i >= GLM53_SPEC_VTRIM_TAU); rows of drafts > n* are dead (rejection sampler placeholders + routed\n"
           "#   experts skipped). shadow only logs. Stats line '[glm53-spec-vtrim] rank R mode ...' every\n"
           "#   GLM53_SPEC_VTRIM_LOG verify steps (first after 64). On BOTH ranks. docs/SPEC_VTRIM.md.\n") + SV4_OLD
SV_EDITS = (("SV1 validation", SV1_OLD, SV1_NEW, 1), ("SV2 head -e", SV2_OLD, SV2_NEW, 1),
            ("SV3 worker list", SV3_OLD, SV3_NEW, 1), ("SV4 note", SV4_OLD, SV4_NEW, 1))


def add_specvtrim(s: str) -> tuple[str, str]:
    """the vtrim (r16z6) stage -> + SV1-SV4 (the GLM53_SPEC_VTRIM knobs), with the same kind of checks."""
    if re.search(rf"\b{SV_KNOB}\b", s):
        sys.exit(f"ABORT: {SV_KNOB} already in the stage")
    for name, old, _new, n in SV_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: specvtrim {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in SV_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the specvtrim edits: {r.stderr.decode()}")
    for k in SV_NAMES:
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_SPEC_VTRIM must be unset/empty/off/0, shadow or on/1")
    sit = s2.count("site/glm53_spec_vtrim.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {SV_KNOB} validation x{val} (need 1), site check x{sit} (need 2)")
    back = s2
    for _name, old, new, n in SV_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the specvtrim stage differs from the incoming stage outside SV1-SV4")
    return s2, f"+ specvtrim SV1-SV4 ({' + '.join(SV_NAMES)}: head -e x1 each, worker list x1 each, validation x1)"


# ---- r16z7 (prefixhit-adv, docs/PREFIX_HIT_TAIL.md): GLM53_KPOOL_TAIL_POSITIONS (0|2) + GLM53_MAMBA_ALIGN_SEED (0|1),
# two bundle overlays run by patch_tf_bundle.py (patch_kpool_tail_positions.py / patch_mamba_align_seed.py), on top of
# the r16z6sv stage (... -> VT -> SV -> PT). Both default to UNSET = production byte for byte (the bundle prints its
# skip line). Value 1 of the tail switch exists in the overlay (positions only) but is REFUSED here: the circular
# slots it builds are a fresh tensor that FULL CUDA graph decode replays never read (production serves every uniform
# spec-verify step as a FULL replay), so =1 would look applied and fix nothing.
PT_KNOB = "GLM53_KPOOL_TAIL_POSITIONS"
PT_NAMES = ("GLM53_KPOOL_TAIL_POSITIONS", "GLM53_MAMBA_ALIGN_SEED")
PT2_OLD = SV2_NEW
PT3_OLD = SV3_NEW
PT4_OLD = SV4_NEW
PT1_OLD = SV1_NEW
PT1_NEW = PT1_OLD + """    # [kpooltail] the kpool indexer tail ring per request (docs/PREFIX_HIT_TAIL.md), on BOTH ranks (a one-rank fix
    # makes the ranks' pooled indexer keys differ = silently degraded sparse top-k). Unset/empty/0 = stock (every
    # decode tail write past the first ring lands in one ring shared by all running requests, and stale warm-up
    # block ids overwrite pooled indexer keys of other requests / cached prefixes); 2 = the overlay
    # patch_kpool_tail_positions.py gives KpoolTailMetadataBuilder the token positions AND writes the circular
    # per-request slots into the persistent slot-mapping buffer (graph-safe). 1 is refused: it builds the slots in
    # a fresh tensor that FULL CUDA graph decode replays never read.
    if [ -n "${GLM53_KPOOL_TAIL_POSITIONS:-}" ]; then
        case "$GLM53_KPOOL_TAIL_POSITIONS" in
            0|2) ;;
            1) echo "GLM53_KPOOL_TAIL_POSITIONS=1 is refused: FULL CUDA graph decode replays do not read it; use 2 (docs/PREFIX_HIT_TAIL.md)" >&2; return 2;;
            *) echo "GLM53_KPOOL_TAIL_POSITIONS must be unset/empty, 0 or 2 (value not printed)" >&2; return 2;;
        esac
    fi
    # [mambaseed] a prefix-hit / resumed request seeds its KDA running column in mamba blocks
    # (cache_config.mamba_block_size) instead of the engine core's recomputed cache_config.block_size; identical in
    # today's multiprocess workers (hardening). Unset/empty/0 = stock; 1 = the overlay patch_mamba_align_seed.py.
    if [ -n "${GLM53_MAMBA_ALIGN_SEED:-}" ]; then
        _glm53_validate_bool_flag GLM53_MAMBA_ALIGN_SEED "$GLM53_MAMBA_ALIGN_SEED" || return
    fi
    for __pt_kv in GLM53_KPOOL_TAIL_POSITIONS:2:patch_kpool_tail_positions.py GLM53_MAMBA_ALIGN_SEED:1:patch_mamba_align_seed.py; do
        __pt_n="${__pt_kv%%:*}"; __pt_v="${__pt_kv#*:}"; __pt_v="${__pt_v%%:*}"; __pt_f="${__pt_kv##*:}"
        if [ "$(eval "echo \\${$__pt_n:-}")" = "$__pt_v" ]; then
            if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/$__pt_f" ]; then
                echo "$__pt_n=$__pt_v requires $TF_BUNDLE_DIR_HOST/overlay/$__pt_f (the bundle overlay this kit ships)" >&2
                return 2
            fi
            if ! grep -qF "(\\"$__pt_f\\", \\"$__pt_n\\")" "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
                echo "$__pt_n=$__pt_v requires $TF_BUNDLE_PATCH_HOST to run $__pt_f (the r16z7 patch_tf_bundle.py)" >&2
                return 2
            fi
        fi
    done
"""
PT2_NEW = PT2_OLD + "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in PT_NAMES)
PT3_NEW = PT3_OLD.replace(" GLM53_SPEC_VTRIM_LOG; do  # [tf-exl3-fork]",
                          " GLM53_SPEC_VTRIM_LOG " + " ".join(PT_NAMES) + "; do  # [tf-exl3-fork]")
PT4_NEW = ("# [kpooltail] GLM53_KPOOL_TAIL_POSITIONS=2 (unset/empty/0 = stock; 1 refused) -> the bundle overlay\n"
           "#   patch_kpool_tail_positions.py: per-request circular kpool tail slots, written into the persistent\n"
           "#   slot-mapping buffer (FULL CUDA graph safe) - stock shares one tail ring between all running requests\n"
           "#   and lets stale warm-up block ids overwrite pooled indexer keys. [mambaseed] GLM53_MAMBA_ALIGN_SEED=1 ->\n"
           "#   patch_mamba_align_seed.py: prefix-hit KDA seed column in mamba blocks (hardening). On BOTH ranks.\n"
           "#   docs/PREFIX_HIT_TAIL.md.\n") + PT4_OLD
PT_EDITS = (("PT1 validation", PT1_OLD, PT1_NEW, 1), ("PT2 head -e", PT2_OLD, PT2_NEW, 1),
            ("PT3 worker list", PT3_OLD, PT3_NEW, 1), ("PT4 note", PT4_OLD, PT4_NEW, 1))


def add_kpool_tail(s: str) -> tuple[str, str]:
    """the specvtrim (r16z6sv) stage -> + PT1-PT4 (GLM53_KPOOL_TAIL_POSITIONS + GLM53_MAMBA_ALIGN_SEED)."""
    for k in PT_NAMES:
        if re.search(rf"\b{k}\b", s):
            sys.exit(f"ABORT: {k} already in the stage")
    for name, old, _new, n in PT_EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: kpooltail {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in PT_EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the kpooltail edits: {r.stderr.decode()}")
    for k in PT_NAMES:
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        m = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = m.group(1).replace("\\\n", " ").split().count(k) if m else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_KPOOL_TAIL_POSITIONS must be unset/empty, 0 or 2")
    one = s2.count("GLM53_KPOOL_TAIL_POSITIONS=1 is refused")
    seed = s2.count('_glm53_validate_bool_flag GLM53_MAMBA_ALIGN_SEED "$GLM53_MAMBA_ALIGN_SEED"')
    if (val, one, seed) != (1, 1, 1):
        sys.exit(f"ABORT: kpooltail validation x{val}, =1 refusal x{one}, seed validation x{seed} (need 1 each)")
    back = s2
    for _name, old, new, n in PT_EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: the kpooltail stage differs from the incoming stage outside PT1-PT4")
    return s2, f"+ kpooltail PT1-PT4 ({' + '.join(PT_NAMES)}: head -e x1 each, worker list x1 each, validation x1; =1 refused)"


# ---- r16z6ar (decode6): GLM53_DEC_AR1SHOT (+ _MAX_KB / _LOG), the decode-size all-reduce of the 2-rank TP group as one
# all-gather + one bf16 add (one network hop instead of the ring's two; bit-identical: docs/DEC_AR1SHOT.md), on top of
# the r16z6 stage. The three knobs default to UNSET = production byte for byte; they are read by the site module
# glm53_ar1shot.py (setup.py py_modules, imported by the site integrate.py in every process, inert when unset).
# The edits anchor on the vtrim (VT) text as SUBSTRINGS and on the worker serve_env_names loop by regex, so the same
# function also applies on top of later stages that only append to VT1/VT2/VT4 (r16z6dl's DL, r16z7's SV/PT).
AR_KNOB = "GLM53_DEC_AR1SHOT"
AR_NAMES = ("GLM53_DEC_AR1SHOT", "GLM53_DEC_AR1SHOT_MAX_KB", "GLM53_DEC_AR1SHOT_LOG")
AR1_BLOCK = """    # [ar1shot] the decode-size all-reduce of the 2-rank TP group as ONE all-gather + one bf16 add (one network hop
    # instead of the ring's two; the bf16 sum is bit-identical, docs/DEC_AR1SHOT.md), on both ranks. Unset/empty/0 =
    # production; 1 = serve the one-shot all-reduce; verify = serve production's and count differing elements (log
    # only, ~+1 collective per all-reduce). _MAX_KB = largest served tensor in KiB (1..16384, default 512 = 64 decode
    # rows), _LOG = stats line period in eager all-reduce calls (non-negative integer, default 2000). Anything else
    # would stop at plugin load inside the bundle, so it is refused here - and a mode needs the bundle site/ module.
    if [ -n "${GLM53_DEC_AR1SHOT:-}" ]; then
        case "$GLM53_DEC_AR1SHOT" in
            0|1|verify) ;;
            *) echo "GLM53_DEC_AR1SHOT must be unset/empty/0, 1 or verify (value not printed)" >&2; return 2;;
        esac
    fi
    if [ -n "${GLM53_DEC_AR1SHOT_MAX_KB:-}" ]; then
        if ! [[ "$GLM53_DEC_AR1SHOT_MAX_KB" =~ ^[0-9]+$ ]] || [ "$GLM53_DEC_AR1SHOT_MAX_KB" -lt 1 ] || [ "$GLM53_DEC_AR1SHOT_MAX_KB" -gt 16384 ]; then
            echo "GLM53_DEC_AR1SHOT_MAX_KB must be an integer in 1..16384 (value not printed)" >&2; return 2
        fi
    fi
    if [ -n "${GLM53_DEC_AR1SHOT_LOG:-}" ] && ! [[ "$GLM53_DEC_AR1SHOT_LOG" =~ ^[0-9]+$ ]]; then
        echo "GLM53_DEC_AR1SHOT_LOG must be a non-negative integer (value not printed)" >&2; return 2
    fi
    case "${GLM53_DEC_AR1SHOT:-}" in
        1|verify) if [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_ar1shot.py" ]; then
            echo "GLM53_DEC_AR1SHOT requires $TF_BUNDLE_DIR_HOST/site/glm53_ar1shot.py (the bundle site/ this kit ships)" >&2
            return 2
        fi;;
    esac
"""
AR2_LINES = "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in AR_NAMES)
AR4_NOTE = ("# [ar1shot] GLM53_DEC_AR1SHOT=1|verify (unset/empty/0 = production) -> the site module glm53_ar1shot.py: a\n"
            "#   2-rank all-reduce of <= GLM53_DEC_AR1SHOT_MAX_KB KiB (default 512) becomes pynccl all_gather + one bf16\n"
            "#   add (one network hop; bit-identical sum); verify serves production's and counts differing elements.\n"
            "#   Boot 'glm53_ar1shot: hooked CudaCommunicator.all_reduce' + 'agreement: every rank ready' + '[glm53-ar1shot]\n"
            "#   rank R serving confirmed'. On BOTH ranks (a one-rank install would hang). docs/DEC_AR1SHOT.md.\n")
_AR_LOOP = re.compile(r"(local -a serve_env_names=\(\)\n    local v\n    for v in )(.*?)(; do  # \[tf-exl3-fork\])", re.S)


def add_ar1shot(s: str) -> tuple[str, str]:
    """an r16z6-family stage -> + AR1-AR4 (the GLM53_DEC_AR1SHOT knobs), with the same kind of checks."""
    if re.search(rf"\b{AR_KNOB}\b", s):
        sys.exit(f"ABORT: {AR_KNOB} already in the stage")
    for name, a in (("AR1 validation (VT1)", VT1_NEW), ("AR2 head -e (VT2)", VT2_NEW), ("AR4 note (VT4)", VT4_NEW)):
        c = s.count(a)
        if c != 1:
            sys.exit(f"ABORT: ar1shot {name}: anchor found {c}x (need 1)")
    ms = list(_AR_LOOP.finditer(s))
    if len(ms) != 1:
        sys.exit(f"ABORT: ar1shot AR3 worker list: serve_env_names loop found {len(ms)}x (need 1)")
    m = ms[0]
    body = m.group(2)
    last = body.rsplit("\n", 1)[-1]
    add = " " + " ".join(AR_NAMES)
    if len(last) + len(add) > 116:
        new_body = body + " \\\n            " + " ".join(AR_NAMES)
    else:
        new_body = body + add
    s2 = s[:m.start(2)] + new_body + s[m.end(2):]
    s2 = s2.replace(VT1_NEW, VT1_NEW + AR1_BLOCK, 1).replace(VT2_NEW, VT2_NEW + AR2_LINES, 1).replace(
        VT4_NEW, AR4_NOTE + VT4_NEW, 1)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the ar1shot edits: {r.stderr.decode()}")
    for k in AR_NAMES:
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        mm = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = mm.group(1).replace("\\\n", " ").split().count(k) if mm else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_DEC_AR1SHOT must be unset/empty/0, 1 or verify")
    sit = s2.count("site/glm53_ar1shot.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {AR_KNOB} validation x{val} (need 1), site check x{sit} (need 2)")
    back = s2.replace(AR1_BLOCK, "", 1).replace(AR2_LINES, "", 1).replace(AR4_NOTE, "", 1)
    back = back.replace(new_body, body, 1)
    if back != s:
        sys.exit("ABORT: the ar1shot stage differs from the incoming stage outside AR1-AR4")
    return s2, f"+ ar1shot AR1-AR4 ({' + '.join(AR_NAMES)}: head -e x1 each, worker list x1 each, validation x1)"


def apply_ar1shot_file(src: str, dst: str) -> str:
    """decode6 kit derivation: AR1-AR4 on an EXISTING r16z6-family start.sh (the kit's launcher/start.sh); every other byte unchanged (checked by add_ar1shot's reverse replay)."""
    s = Path(src).read_text()
    s2, m = add_ar1shot(s)
    Path(dst).write_text(s2)
    os.chmod(dst, 0o755)
    return m


# ---- r16z6dl (decode5) / r16z8: GLM53_DEC_DLMH (+ _C / _GROUP / _LOG), the DFlash2 drafter's candidate head on a 4-bit
# coarse copy of the lm_head + exact FP8 rescoring of the selected column octets (docs/DEC_DLMH.md). r16z6dl = the r16z6
# stage + DL1-DL4 (... -> KL -> VT -> DL); r16z8 = the r16z7ar stage (r16z7 + AR1-AR4) + DL1-DL4. The four knobs default
# to UNSET = production byte for byte; they are read by the site module glm53_dlmh.py (setup.py py_modules, imported by
# the site integrate.py in every process, inert when unset; its kernel ships as site/tf_dlmh_ext*.so).
# r16z8: the edits anchor on the vtrim (VT) text as SUBSTRINGS and on the worker serve_env_names loop by regex (as
# add_ar1shot), so they apply on top of every r16z6-family stage; on the r16z6 stage they reproduce decode5's r16z6dl
# start.sh byte for byte (the names are appended to the loop's last line, as decode5's DL3 did).
DL_KNOB = "GLM53_DEC_DLMH"
DL_NAMES = ("GLM53_DEC_DLMH", "GLM53_DEC_DLMH_C", "GLM53_DEC_DLMH_GROUP", "GLM53_DEC_DLMH_LOG")
DL1_BLOCK = """    # [dlmh] the DFlash2 drafter's candidate head reads a 4-bit coarse copy of the lm_head and recomputes the exact
    # FP8 logits of the selected column octets (docs/DEC_DLMH.md), on both ranks. Unset/empty/0 = production;
    # verify = serve production's candidates and count where the two-stage head would differ (log only);
    # 1 = serve the two-stage head (byte-identical candidates when the coarse top-C holds every element >= the 16th).
    # _C = column octets per row (multiple of 8 in 16..256, default 128), _GROUP = coarse group size (32|64|128,
    # default 128), _LOG = stats line period in drafter steps (non-negative integer, default 2000). Anything else
    # would stop at plugin load inside the bundle, so it is refused here - and a mode needs the bundle site/ module.
    if [ -n "${GLM53_DEC_DLMH:-}" ]; then
        case "$GLM53_DEC_DLMH" in
            0|1|verify) ;;
            *) echo "GLM53_DEC_DLMH must be unset/empty/0, 1 or verify (value not printed)" >&2; return 2;;
        esac
    fi
    if [ -n "${GLM53_DEC_DLMH_C:-}" ]; then
        if ! [[ "$GLM53_DEC_DLMH_C" =~ ^[0-9]+$ ]] || [ "$GLM53_DEC_DLMH_C" -lt 16 ] || [ "$GLM53_DEC_DLMH_C" -gt 256 ] || [ $((GLM53_DEC_DLMH_C % 8)) -ne 0 ]; then
            echo "GLM53_DEC_DLMH_C must be a multiple of 8 in 16..256 (value not printed)" >&2; return 2
        fi
    fi
    if [ -n "${GLM53_DEC_DLMH_GROUP:-}" ]; then
        case "$GLM53_DEC_DLMH_GROUP" in
            32|64|128) ;;
            *) echo "GLM53_DEC_DLMH_GROUP must be 32, 64 or 128 (value not printed)" >&2; return 2;;
        esac
    fi
    if [ -n "${GLM53_DEC_DLMH_LOG:-}" ] && ! [[ "$GLM53_DEC_DLMH_LOG" =~ ^[0-9]+$ ]]; then
        echo "GLM53_DEC_DLMH_LOG must be a non-negative integer (value not printed)" >&2; return 2
    fi
    case "${GLM53_DEC_DLMH:-}" in
        1|verify) if [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_dlmh.py" ]; then
            echo "GLM53_DEC_DLMH requires $TF_BUNDLE_DIR_HOST/site/glm53_dlmh.py (the bundle site/ this kit ships)" >&2
            return 2
        fi;;
    esac
"""
DL2_LINES = "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in DL_NAMES)
DL4_NOTE = ("# [dlmh] GLM53_DEC_DLMH=1|verify (unset/empty/0 = production) -> the site module glm53_dlmh.py: the drafter's\n"
            "#   top-16 candidates from an int4 coarse lm_head copy + the exact FP8 logits of the coarse top-C column\n"
            "#   octets per row (byte-identical when they hold every element >= the 16th); verify serves production's and\n"
            "#   counts differences. Boot 'glm53_dlmh: rank R self-test: candidates and unary logits byte-equal' + stats\n"
            "#   '[glm53-dlmh] rank R mode ...' every GLM53_DEC_DLMH_LOG drafter steps. On BOTH ranks. docs/DEC_DLMH.md.\n")


def add_dlmh(s: str) -> tuple[str, str]:
    """an r16z6-family stage (r16z6 -> r16z6dl, r16z7ar -> r16z8) -> + DL1-DL4 (the GLM53_DEC_DLMH knobs), with the same
    kind of checks as the other stages (anchor counts, bash -n, head/worker placement, validation count, reverse replay)."""
    if re.search(rf"\b{DL_KNOB}\b", s):
        sys.exit(f"ABORT: {DL_KNOB} already in the stage")
    for name, a in (("DL1 validation (VT1)", VT1_NEW), ("DL2 head -e (VT2)", VT2_NEW), ("DL4 note (VT4)", VT4_NEW)):
        c = s.count(a)
        if c != 1:
            sys.exit(f"ABORT: dlmh {name}: anchor found {c}x (need 1)")
    ms = list(_AR_LOOP.finditer(s))
    if len(ms) != 1:
        sys.exit(f"ABORT: dlmh DL3 worker list: serve_env_names loop found {len(ms)}x (need 1)")
    m = ms[0]
    body = m.group(2)
    add = " " + " ".join(DL_NAMES)
    # appended to the loop's LAST line, never wrapped: decode5's r16z6dl appended the names to the r16z6 loop's (long)
    # VT line - reproduced byte for byte; on r16z7ar the last line is add_ar1shot's short wrapped AR line
    new_body = body + add
    s2 = s[:m.start(2)] + new_body + s[m.end(2):]
    s2 = s2.replace(VT1_NEW, VT1_NEW + DL1_BLOCK, 1).replace(VT2_NEW, VT2_NEW + DL2_LINES, 1).replace(
        VT4_NEW, DL4_NOTE + VT4_NEW, 1)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the dlmh edits: {r.stderr.decode()}")
    for k in DL_NAMES:
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        mm = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = mm.group(1).replace("\\\n", " ").split().count(k) if mm else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_DEC_DLMH must be unset/empty/0, 1 or verify")
    sit = s2.count("site/glm53_dlmh.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {DL_KNOB} validation x{val} (need 1), site check x{sit} (need 2)")
    back = s2.replace(DL1_BLOCK, "", 1).replace(DL2_LINES, "", 1).replace(DL4_NOTE, "", 1)
    back = back.replace(new_body, body, 1)
    if back != s:
        sys.exit("ABORT: the dlmh stage differs from the incoming stage outside DL1-DL4")
    return s2, f"+ dlmh DL1-DL4 ({' + '.join(DL_NAMES)}: head -e x1 each, worker list x1 each, validation x1)"


def apply_dlmh_file(src: str, dst: str) -> str:
    """kit derivation: DL1-DL4 on an EXISTING r16z6-family start.sh (a kit's launcher/start.sh); every other byte unchanged (checked by add_dlmh's reverse replay)."""
    s = Path(src).read_text()
    s2, m = add_dlmh(s)
    Path(dst).write_text(s2)
    os.chmod(dst, 0o755)
    return m


# ---- r16z8p: GLM53_MLA_PLAN_PIN (the sparse-MLA plan's CPU staging page-locked in a 2-slot ring, docs/MLA_PLAN_PIN.md)
# + persistent ABLIT (start.sh honours an .env ABLIT; caller export still wins; GLM53_MODEL_PRESET=abliterated still
# forces 0). PP1-PP4 anchor like add_ar1shot / add_dlmh (VT text as substrings + the worker loop by regex); the knob
# defaults to UNSET = production byte for byte (the site module glm53_mla_planpin.py, imported by integrate.py in every
# process, installs nothing). AB1/AB2 edit the .env preamble and the abliteration block of the APC start.sh.
PP_KNOB = "GLM53_MLA_PLAN_PIN"
PP_NAMES = ("GLM53_MLA_PLAN_PIN",)
PP1_BLOCK = """    # [planpin] production's sparse-MLA plan (_SM90State.plan) stages its indptr / lens in PAGE-LOCKED memory (a 2-slot
    # ring with a per-slot CUDA event and int workspace) instead of pageable tensors whose 65540 B indptr copy at
    # max-num-batched-tokens 16384 blocks the host until the drafter graph drained (docs/MLA_PLAN_PIN.md), on both
    # ranks. Unset/empty/0 = production; 1 = pinned staging (same device bytes, no host stall). Anything else would
    # only log a refusal inside the bundle, so it is refused here - and 1 needs the bundle site/ module.
    if [ -n "${GLM53_MLA_PLAN_PIN:-}" ]; then
        case "$GLM53_MLA_PLAN_PIN" in
            0|1) ;;
            *) echo "GLM53_MLA_PLAN_PIN must be unset/empty, 0 or 1 (value not printed)" >&2; return 2;;
        esac
    fi
    if [ "${GLM53_MLA_PLAN_PIN:-}" = 1 ] && [ ! -s "$TF_BUNDLE_DIR_HOST/site/glm53_mla_planpin.py" ]; then
        echo "GLM53_MLA_PLAN_PIN requires $TF_BUNDLE_DIR_HOST/site/glm53_mla_planpin.py (the bundle site/ this kit ships)" >&2
        return 2
    fi
"""
PP2_LINES = "".join(f'        -e {k}="${{{k}:-}}" \\\n' for k in PP_NAMES)
PP4_NOTE = ("# [planpin] GLM53_MLA_PLAN_PIN=1 (unset/empty/0 = production) -> the site module glm53_mla_planpin.py: the\n"
            "#   sparse-MLA plan's staging is page-locked (2-slot ring + per-slot event + int workspace), so the 65540 B\n"
            "#   indptr copy no longer blocks the host on the drafter graph; same device bytes. Boot 'glm53_mla_planpin:\n"
            "#   patched _SM90State.plan' + 'rank R self-test: 3/3 device plan buffers == pinned staging' + '[glm53-mla-\n"
            "#   planpin] rank R serving confirmed'. On BOTH ranks. docs/MLA_PLAN_PIN.md.\n")


def add_planpin(s: str) -> tuple[str, str]:
    """the r16z8 stage -> + PP1-PP4 (GLM53_MLA_PLAN_PIN), with the same checks as add_ar1shot / add_dlmh."""
    if re.search(rf"\b{PP_KNOB}\b", s):
        sys.exit(f"ABORT: {PP_KNOB} already in the stage")
    for name, a in (("PP1 validation (VT1)", VT1_NEW), ("PP2 head -e (VT2)", VT2_NEW), ("PP4 note (VT4)", VT4_NEW)):
        c = s.count(a)
        if c != 1:
            sys.exit(f"ABORT: planpin {name}: anchor found {c}x (need 1)")
    ms = list(_AR_LOOP.finditer(s))
    if len(ms) != 1:
        sys.exit(f"ABORT: planpin PP3 worker list: serve_env_names loop found {len(ms)}x (need 1)")
    m = ms[0]
    body = m.group(2)
    last = body.rsplit("\n", 1)[-1]
    add = " " + " ".join(PP_NAMES)
    new_body = body + (" \\\n            " + " ".join(PP_NAMES) if len(last) + len(add) > 116 else add)
    s2 = s[:m.start(2)] + new_body + s[m.end(2):]
    s2 = s2.replace(VT1_NEW, VT1_NEW + PP1_BLOCK, 1).replace(VT2_NEW, VT2_NEW + PP2_LINES, 1).replace(
        VT4_NEW, PP4_NOTE + VT4_NEW, 1)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the planpin edits: {r.stderr.decode()}")
    for k in PP_NAMES:
        head = s2.count(f'        -e {k}="${{{k}:-}}" \\\n')
        mm = re.search(r"local -a serve_env_names=\(\)\n    local v\n    for v in (.*?); do", s2, re.S)
        worker = mm.group(1).replace("\\\n", " ").split().count(k) if mm else 0
        if (head, worker) != (1, 1):
            sys.exit(f"ABORT: {k} placement head={head} worker={worker} (need 1, 1)")
    val = s2.count("GLM53_MLA_PLAN_PIN must be unset/empty, 0 or 1")
    sit = s2.count("site/glm53_mla_planpin.py")
    if (val, sit) != (1, 2):
        sys.exit(f"ABORT: {PP_KNOB} validation x{val} (need 1), site check x{sit} (need 2)")
    back = s2.replace(PP1_BLOCK, "", 1).replace(PP2_LINES, "", 1).replace(PP4_NOTE, "", 1)
    back = back.replace(new_body, body, 1)
    if back != s:
        sys.exit("ABORT: the planpin stage differs from the incoming stage outside PP1-PP4")
    return s2, f"+ planpin PP1-PP4 ({' + '.join(PP_NAMES)}: head -e x1, worker list x1, validation x1)"


# AB1: the .env preamble. Today start.sh CLEARS an .env ABLIT right after sourcing .env (ABLIT=0) and honours only a
# caller-exported ABLIT (re-applied by the override loop that follows). r16z8p remembers whether the CALLER exported
# ABLIT (before .env is sourced), keeps an .env value (validated 0|1) when the caller did not, and otherwise sets 0:
#   caller exported (any value, also empty)  -> the caller's value (the loop below re-exports it), as today
#   no caller export, .env ABLIT=0|1          -> the .env value                                   (NEW)
#   no caller export, no / empty .env ABLIT   -> 0                                                 (as today)
AB1_OLD = """set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
# Stock o_proj unless the caller exported ABLIT. Only this flag is cleared;
# every other .env knob stays, including GLM53_APC_RETENTION_INTERVAL_SWA
# from #207. Use ABLIT=1 ./start.sh to opt in.
ABLIT=0
"""
AB1_NEW = """# [r16z8p ablit] whether the CALLER exported ABLIT (.env is not sourced yet: only an export can be set here)
_glm53_ablit_caller="${ABLIT+x}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
# [r16z8p ablit] ABLIT from .env is honoured (owner decision 2026-10-05: production runs ablit, persistently): a
# caller export still wins (re-applied by the override loop below, explicit empty included), an .env value must be
# 0 or 1, no / empty .env value = stock o_proj (0) as before; GLM53_MODEL_PRESET=abliterated still forces 0 below.
# Every other .env knob stays, including GLM53_APC_RETENTION_INTERVAL_SWA from #207.
if [ -n "$_glm53_ablit_caller" ]; then
    _glm53_ablit_src="caller export"
elif [ -n "${ABLIT:-}" ]; then
    case "$ABLIT" in
        0|1) _glm53_ablit_src=".env" ;;
        *) die "ABLIT in .env must be 0 or 1 (value not printed)" ;;
    esac
else
    ABLIT=0
    _glm53_ablit_src="default (no ABLIT in .env)"
fi
"""
# AB2: after the abliteration block's preset override: log the EFFECTIVE value and its source (one line per start)
AB2_OLD = """[ "$GLM53_MODEL_PRESET" = "abliterated" ] && ABLIT=0
"""
AB2_NEW = """[ "$GLM53_MODEL_PRESET" = "abliterated" ] && ABLIT=0
# [r16z8p ablit] the effective value and where it came from (boot_checks compares the two containers' ABLIT)
[ "$GLM53_MODEL_PRESET" = "abliterated" ] && _glm53_ablit_src="${_glm53_ablit_src}; forced 0 by GLM53_MODEL_PRESET=abliterated"
log "ablit: effective ABLIT=$ABLIT (source: ${_glm53_ablit_src:-default})"
unset _glm53_ablit_caller _glm53_ablit_src
"""


def add_ablit_env(s: str) -> tuple[str, str]:
    """AB1 + AB2 (persistent ABLIT) on an r16z8-family stage; anchor counts, bash -n, reverse replay."""
    for name, a in (("AB1 .env preamble", AB1_OLD), ("AB2 preset override", AB2_OLD)):
        c = s.count(a)
        if c != 1:
            sys.exit(f"ABORT: ablit {name}: anchor found {c}x (need 1)")
    if "_glm53_ablit_caller" in s:
        sys.exit("ABORT: ablit edits already in the stage")
    s2 = s.replace(AB1_OLD, AB1_NEW, 1).replace(AB2_OLD, AB2_NEW, 1)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n after the ablit edits: {r.stderr.decode()}")
    if s2.index("_glm53_ablit_caller=") > s2.index('source "$SCRIPT_DIR/.env"') or \
            s2.index('source "$SCRIPT_DIR/.env"') > s2.index("for _kv in ${_caller_overrides[@]+"):
        sys.exit("ABORT: ablit AB1 order (caller check -> source .env -> override loop) broken")
    if s2.count("\nABLIT=0\n") != 0 or s2.count('log "ablit: effective ABLIT=') != 1:
        sys.exit("ABORT: ablit: the unconditional ABLIT=0 survived or the effective-value log is not exactly once")
    back = s2.replace(AB1_NEW, AB1_OLD, 1).replace(AB2_NEW, AB2_OLD, 1)
    if back != s:
        sys.exit("ABORT: the ablit stage differs from the incoming stage outside AB1-AB2")
    return s2, "+ ablit AB1-AB2 (ABLIT from .env honoured when the caller did not export it, 0|1 validated; caller export and GLM53_MODEL_PRESET=abliterated precedence kept; effective value logged)"


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--apply-dlmh":
        print(apply_dlmh_file(sys.argv[2], sys.argv[3]))
        sys.exit(0)
    sys.exit(main())
