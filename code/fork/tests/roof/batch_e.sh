#!/usr/bin/env bash
# inside the container (/w): regression of the modules this branch touches (fp8_gemv.py hook call, integrate.py
# plugin_register), AOT extensions, launcher overlay bound; logs -> tests/logs/roof/reg_*.log
cd /w
L=tests/logs/roof
python3 -c "import sys, torch; sys.path.insert(0, '/w'); import tf_fp8_roof_ext as r, tf_fp8_gemv_ext as f; assert r.VERSION == 1 and f.VERSION == 1; print('AOT', r.__file__, 'VERSION', r.VERSION)" > $L/reg_aot.log 2>&1; echo "aot rc $?"
for t in test_fp8_roof test_fp8_gemv test_fp8_integrate test_fp8_large_m_integrate test_integrate; do
  python3 -u tests/$t.py > $L/reg_$t.log 2>&1; echo "$t rc $? $(grep -E '^checks:|^RESULT' $L/reg_$t.log | tr '\n' ' ')"
done
