import sys, torch
run, L = sys.argv[1], sys.argv[2]
for arm in sys.argv[3].split(","):
    T = torch.load(f"{run}/phtrace_L{L}_{arm}.pt", weights_only=False); D = torch.load(f"{run}/phdump_L{L}_{arm}.pt", weights_only=False)
    out = []
    for name in sorted(T["kda"], key=lambda n: int(n.split(".")[2]))[:3]:
        t = T["kda"][name]
        v1 = D["views"][name + "[1]"]; v0 = D["views"][name + "[0]"]
        ck = v1["data"][0].float(); dst = v1["data"][1].float()          # checkpoint column / destination column (pre-step)
        r = t["rec"]
        z = bool((r == 0).all())
        out.append(f"{name.split('.')[2]}: slot {t['slot']} ids(ck,dst)={v1['ids']} rec==ckpt {bool(torch.equal(r, ck))} rec==dst_prestep {bool(torch.equal(r, dst))} rec_zero {z} "
                   f"conv==ckpt {bool(torch.equal(t['conv'][:3], v0['data'][0].float()[:3]))} has_init {t['has_init'].tolist() if t['has_init'] is not None else None} "
                   f"np/nd {t['num_prefills']}/{t['num_decodes']}")
    print(arm, "computed", T["computed"], "n", T["n"]); print("   " + "\n   ".join(out))
