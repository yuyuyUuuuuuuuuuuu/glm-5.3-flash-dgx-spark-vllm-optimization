"""[dec-hostloop] print the source fingerprints of the production functions glm53_hostloop relies on."""
import importlib
import sys
sys.path.insert(0, "/w")
import glm53_hostloop as H

mr = importlib.import_module(H.MR_MODULE)
sm90 = importlib.import_module(H.SM90_MODULE)
ok, why, fps = H._verify_sources(mr, sm90)
for k, v in sorted(fps.items()):
    print("FP %-48s %s" % (k, v))
print("verified:", ok, why)
