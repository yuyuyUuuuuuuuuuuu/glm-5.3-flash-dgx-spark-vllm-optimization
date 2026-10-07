"""opt-moe default-path identity: with OPTMOE_PREV=1 the production kit's (r16z2) glm53_moe_e4m3 module + extension
(copied into tests/optmoe/prev/, untracked) compute the fixed-input outputs, else this branch's; OPTMOE_SAVE=path saves,
OPTMOE_CMP=path compares (rel-L2 and the fraction of differing bf16 elements after the cast, vs the fp32 atomics-order
spread of two runs of the same build). Cases: production shapes, real/collapsed routing, default and f16-down variants.
Run twice: OPTMOE_PREV=1 OPTMOE_SAVE=tests/optmoe/prev/ref.pt ..., then OPTMOE_CMP=tests/optmoe/prev/ref.pt ..."""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    if os.environ.get("OPTMOE_PREV") == "1":
        load("glm53_moe_e4m3_ext", os.path.join(HERE, "prev", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"))
        M = load("glm53_moe_e4m3", os.path.join(HERE, "prev", "glm53_moe_e4m3_prod.py"))
        print("using the production kit's module + extension (tests/optmoe/prev)", flush=True)
    else:
        import glm53_moe_e4m3 as M
        print(f"using this branch's module ({M.__file__}) + extension ({M._ext().__file__})", flush=True)
    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    save, cmp = os.environ.get("OPTMOE_SAVE"), os.environ.get("OPTMOE_CMP")
    ref = torch.load(cmp) if cmp else None
    res = {}
    for T, kind, seed, var in ((13824, "real", 41, 0), (13856, "collapsed", 42, 0), (4289, "real", 43, 0),
                               (300, "real", 44, 0), (13824, "real", 45, 16), (4289, "collapsed", 46, 16)):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        ids = C.routing(kind, T, seed, dev)
        w = C.weights_for(T, seed, dev).float()
        tag = f"T={T} {kind} v{var}"
        o = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": var}).clone()
        o2 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": var}).clone()
        aa = float((o - o2).double().norm() / o.double().norm())
        res[tag] = o.cpu()
        print(f"  [{tag}] dtype {o.dtype}, run-to-run rel-L2 {aa:.2e}", flush=True)
        if ref is not None:
            r = ref[tag]
            d = float((o.cpu() - r).double().norm() / r.double().norm())
            fb = float((o.cpu().to(torch.bfloat16) != r.to(torch.bfloat16)).float().mean())
            fb2 = float((o.to(torch.bfloat16) != o2.to(torch.bfloat16)).float().mean())
            print(f"  [{tag}] vs saved: rel-L2 {d:.2e}; bf16 elements differing {fb:.2e} (run-to-run {fb2:.2e})",
                  flush=True)
            CHK(o.dtype == torch.float32 and d < 1e-6 and fb < 10 * max(fb2, 1e-5),
                f"[{tag}] default path == production kit's within the fp32-atomics order class ({d:.2e}, {fb:.2e})")
    if save:
        torch.save(res, save)
        print(f"saved {len(res)} outputs to {save}")
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
