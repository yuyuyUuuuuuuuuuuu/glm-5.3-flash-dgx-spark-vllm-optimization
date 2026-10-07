#!/usr/bin/env python3
"""Japanese prose decode bench: same as tests/bench_decode.py prose mode, with a Japanese prompt."""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("bench_decode", os.path.join(os.environ.get("LAUNCHER_DIR") or os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"), "tests/bench_decode.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.BENCH_PROMPT = "ハッシュマップの仕組みを、衝突の扱い、リサイズ、計算量を含めて、日本語で順を追って詳しく説明してください。"
sys.exit(m.main())
