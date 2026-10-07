#!/usr/bin/env python3
"""FKDA review check (host, no torch): production runs GLM53_PREFILL_QUICKWINS=all
(env.r16), whose kda_conv item patches Glm5NextLinearAttention._forward only if
that function's AST fingerprint is in glm53_prefill_quickwins.VERIFIED and its
KDA_CONV edit anchor occurs exactly once. patch_flashkda.py edits the same
function on disk before the plugin loads, so with GLM53_KDA_FLASHKDA=1 the
fingerprint must still be accepted, or kda_conv is refused ("NOT installed",
boot_checks.sh MISS) and production loses that quick win.

Usage: check_quickwins_compat.py <image kda.py>   (e.g. extracted with
  docker run --rm --entrypoint cat <image> .../vllm/models/glm5next/nvidia/kda.py)
Exit 0 = compatible, 1 = kda_conv would be refused under FlashKDA.
"""
import ast
import hashlib
import importlib.util
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fingerprint(src: str, cls: str = "Glm5NextLinearAttention", fn: str = "_forward") -> str:
    """inspect.getsource-equivalent (decorators included), the quickwins recipe:
    sha256(ast.dump(ast.parse(dedent(source))))[:16]."""
    lines = src.splitlines(keepends=True)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == fn:
                    start = min([d.lineno for d in m.decorator_list] + [m.lineno])
                    seg = "".join(lines[start - 1:m.end_lineno])
                    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(seg))).encode()).hexdigest()[:16]
    raise SystemExit(f"{cls}.{fn} not found")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_verdict(qw_text: str):
    ns: dict = {}
    exec(qw_text[qw_text.index("VERIFIED = {"):qw_text.index("STATS = {")], ns)
    return ns["VERIFIED"][("kda_conv", "Glm5NextLinearAttention._forward")]


def main() -> int:
    stock = Path(sys.argv[1]).read_text()
    pf = load(REPO / "overlay/patch_flashkda.py", "patch_flashkda")
    qw_path = REPO / "glm53_prefill_quickwins.py"
    qw_src = qw_path.read_text()
    allowed = extract_verdict(qw_src)
    ns: dict = {}
    exec(qw_src[qw_src.index("KDA_CONV = ("):qw_src.index("CONV_OUT_SIG")], ns)
    kda_conv_old, kda_conv_new = ns["KDA_CONV"][0], ns["KDA_CONV"][1]
    patched = pf.prepare(stock)
    fs, fp_ = fingerprint(stock), fingerprint(patched)
    anchor = patched.count(kda_conv_old)
    print(f"stock   _forward fingerprint {fs} (in VERIFIED: {fs in allowed})")
    print(f"flashkda _forward fingerprint {fp_} (in VERIFIED: {fp_ in allowed}); KDA_CONV anchor count {anchor}")
    if fs not in allowed:
        print("FAIL: the recipe does not reproduce the stock fingerprint (wrong kda.py?)")
        return 2
    ok = True
    # 1. the composition proof: the KDA_CONV edit applies to the FlashKDA-patched
    #    _forward (anchors once, compiles) -- both edits land in the shipped tree
    if anchor != 1:
        print("FAIL: the KDA_CONV anchor does not anchor exactly once on the FlashKDA-patched kda.py")
        ok = False
    else:
        try:
            compile(patched.replace(kda_conv_old, kda_conv_new, 1), "kda.py+quickwins", "exec")
            print("OK: KDA_CONV composes with the FlashKDA patch (edit applies, source compiles)")
        except SyntaxError as exc:
            print(f"FAIL: KDA_CONV does not compile on the FlashKDA-patched kda.py: {exc}")
            ok = False
    # 2. the allowlist extension (what keeps kda_conv installable with =1): run the
    #    patcher's own extend_quickwins_allowlist against a TEMP COPY of the module
    with tempfile.TemporaryDirectory() as td:
        tmp_qw = Path(td) / "glm53_prefill_quickwins.py"
        tmp_qw.write_text(qw_src)
        try:
            act = pf.extend_quickwins_allowlist(fs, fp_, str(tmp_qw))
        except SystemExit as exc:
            print(f"FAIL: extend_quickwins_allowlist refused: {exc}")
            return 1
        allowed_ext = extract_verdict(tmp_qw.read_text())
        print(f"extend_quickwins_allowlist: {act}; VERIFIED kda_conv now "
              f"{sorted(allowed_ext)} (stock kept: {fs in allowed_ext})")
        # 3. idempotent second run
        act2 = pf.extend_quickwins_allowlist(fs, fp_, str(tmp_qw))
        same = extract_verdict(tmp_qw.read_text()) == allowed_ext
        print(f"second run: {act2}, set unchanged: {same}")
        if fp_ not in allowed_ext or fs not in allowed_ext or act2 != "already present" or not same:
            print("FAIL: the extended VERIFIED set does not accept the FlashKDA-patched _forward")
            ok = False
    if not ok:
        return 1
    print("OK: kda_conv installs on the FlashKDA-patched kda.py (fingerprint extended by patch_flashkda.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
