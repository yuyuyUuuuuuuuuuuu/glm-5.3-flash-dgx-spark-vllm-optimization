import json, sys, torch
run, vname, row = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = torch.load(run + "/records.pt", weights_only=False)["records"]
prev = None
for x in rec:
    h = int(x["view_hash"][vname][row])
    if h != prev:
        print(f"step {x['step']:3d} {'P' if x['prefill'] else 'd'} sched {x['sched']} computed {x['computed']} -> hash {h & 0xffffffff:08x}")
        prev = h
