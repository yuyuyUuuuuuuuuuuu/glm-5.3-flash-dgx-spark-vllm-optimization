"""OPTMOE_EXT=<path to a glm53_moe_e4m3_ext .so>: load that build instead of overlay/'s (A/B of builds in progress)."""
import importlib.util
import os
import sys


def preload():
    so = os.environ.get("OPTMOE_EXT")
    if not so:
        return None
    spec = importlib.util.spec_from_file_location("glm53_moe_e4m3_ext", so)
    m = importlib.util.module_from_spec(spec)
    sys.modules["glm53_moe_e4m3_ext"] = m
    spec.loader.exec_module(m)
    print(f"OPTMOE_EXT: using {so}", flush=True)
    return m
