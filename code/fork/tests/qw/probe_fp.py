"""Print the fingerprints of the production functions glm53_prefill_quickwins patches and check its edit anchors."""
import importlib, sys, inspect, textwrap
sys.path.insert(0, "/w")
import glm53_prefill_quickwins as Q
mods = {}
for item, entries in Q.PLAN.items():
    for modname, qual, edits in entries:
        m = mods.get(modname) or importlib.import_module(modname)
        mods[modname] = m
        owner, name, fn = Q._get_attr(m, qual)
        src = inspect.getsource(fn)
        print(item, modname, qual, "fp", Q.source_fingerprint(fn), "anchors", [src.count(o) for o, _ in edits],
              "file", inspect.getsourcefile(fn))
conv = importlib.import_module(Q.M_CONV)
src = textwrap.dedent(inspect.getsource(conv.causal_conv1d_fn))
print("kda_conv", Q.M_CONV, "causal_conv1d_fn fp", Q.source_fingerprint(conv.causal_conv1d_fn), "anchors",
      [src.count(o) for o, _ in (Q.CONV_OUT_SIG, Q.CONV_OUT_ALLOC)])
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, "cuda", torch.version.cuda)
