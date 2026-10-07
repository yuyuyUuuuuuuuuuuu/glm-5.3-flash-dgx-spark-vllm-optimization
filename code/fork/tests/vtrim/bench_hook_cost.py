"""Cost of GLM53_SPEC_VTRIM's MoE hook when nothing is dead: the masked_fill of the routed ids (one tiny kernel per
MoE layer inside the decode graph) on production's decode MoE layer (tests/moeglue_rig.py, 42 layers per graph, corr40,
T = 5 and 8 rows). Modes: base (production apply) vs hook (LIVE all ones, ids.masked_fill(LIVE == 0, -1) first).
Usage: tests/gpu_run.sh python3 -u tests/vtrim/bench_hook_cost.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
import harness as H  # noqa: E402
import moeglue_rig as MR  # noqa: E402


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl(); prod = H.load_prod(); H.load_tf()
    import integrate
    dev = torch.device("cuda", 0)
    integrate.install(prodmod=prod, ext=xl, force=True)
    layers = MR.make_layers(prod, dev, 3, 42)
    rig = MR.Rig(prod, layers, dev)
    live = torch.ones(1024, dtype=torch.uint8, device=dev)
    orig = prod.apply_exl3_experts
    def hooked(x, ids, w, layer):
        b = x.shape[0]
        ids = ids.masked_fill(live[:b].unsqueeze(1) == 0, -1)
        return orig(x, ids, w, layer, limit=MR.LIMIT)
    for T in (5, 8):
        sets = MR.make_sets("corr40", T, 42, 1, dev, seed=99 + T)
        graphs = {}
        for name, ap in (("base", None), ("hook", hooked)):
            ss = [(li, x.clone(), ids.clone(), w.clone()) for (li, x, ids, w) in sets] if name == "hook" else sets
            fns = [lambda s=s, ap=ap: rig.call(s[0], s[1], s[2], s[3], apply=ap) for s in ss]
            graphs[name], _ = MR.capture(fns, dev)
        res = MR.ab_rounds(graphs, 42, 101, 1)
        b, h = MR.med(res["base"]), MR.med(res["hook"])
        print(f"T={T}: base {b:.1f} us/layer, hook {h:.1f} us/layer -> {h - b:+.2f} us/layer, "
              f"{(h - b) * 42 / 1000:+.3f} ms/step (42 MoE layers)", flush=True)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
