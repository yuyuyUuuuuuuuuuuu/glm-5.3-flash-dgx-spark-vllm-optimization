import torch
d = {k: torch.load(f"/w/tests/optmoe_rev/prodso_{k}.pt") for k in ("new", "kit", "new2", "kit2")}
for key in d["new"]:
    a = d["new"][key]
    print(key, {k: int((v[key] != a).sum()) for k, v in d.items()}, "of", a.numel())
