"""Which production quantization/exl3.py this container imports, and whether this fork was tested against it.

Exit 1 when the file is not in integrate.KNOWN_PROD_MODULES. To certify a new production file (docs/PRODUCTION_PLAN.md
Phase 0 item 1): add its sha256 to KNOWN_PROD_MODULES, then tests/run_all.sh and tests/drafter/run_drafter_tests.sh
with the same GPU_RUN_BIND must both pass; commit, then build and install that commit (the startup log then names the
version instead of warning "not a version this fork was tested against").
"""
import sys

sys.path.insert(0, "/w")
import torch  # noqa: F401,E402 - vLLM's import order

import integrate  # noqa: E402
import vllm.model_executor.layers.quantization.exl3 as q  # noqa: E402

i = integrate.prod_identity(q)
ok, why = integrate.k2_compatibility(q)
print("production module", i["file"], "sha256", i["sha256"])
print("known as:", i["known"], "| K2 verified:", ok, why)
if not i["known"]:
    print(f'NOT in integrate.KNOWN_PROD_MODULES. To certify this file: add "{i["sha256"]}": "<label>" to '
          "KNOWN_PROD_MODULES (integrate.py); tests/run_all.sh and tests/drafter/run_drafter_tests.sh with the same "
          "GPU_RUN_BIND must then pass; commit, build and install that commit (docs/PRODUCTION_PLAN.md Phase 0 item 1)")
if not ok:
    print("K2 is not verified for this file: install() will serve decode through the exl3_moe dispatcher only "
          "(WARNING 'K2 apply path NOT installed'); to verify K2, tests/test_apply_fused.py must pass against it and "
          "its fingerprints be added to integrate.K2_VERIFIED_FINGERPRINTS")
sys.exit(0 if i["known"] else 1)
