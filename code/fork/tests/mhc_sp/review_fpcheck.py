import ast, hashlib, textwrap, sys, re
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[2].as_posix())
d = sys.argv[1]
NV = "vllm/models/glm5next/nvidia"
src = open(f"{d}/{NV}/model.py").read()
def fsrc(cls, fn):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for s in node.body:
                if isinstance(s, ast.FunctionDef) and s.name == fn:
                    # inspect.getsource includes decorators; none here
                    return "\n".join(src.split("\n")[s.lineno-1:s.end_lineno]) + "\n"
def fp(s): return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(s))).encode()).hexdigest()[:16]
qwns = {}; exec(compile(open(f"{d}/glm53_prefill_quickwins.py").read().split("STATS = {")[0].split("VERIFIED = {",1)[0] , "q", "exec"), qwns) if False else None
qsrc = open(f"{d}/glm53_prefill_quickwins.py").read()
msrc = open(f"{d}/glm53_moeglue.py").read()
# evaluate the VERIFIED / WARM_VERIFIED literals
def lit(text, name):
    m = re.search(r"^%s = \{.*?^\}" % name, text, re.S | re.M)
    return eval(m.group(0).split("=",1)[1])
V = lit(qsrc, "VERIFIED"); W = lit(msrc, "WARM_VERIFIED")
layer = fsrc("Glm5NextDecoderLayer", "forward"); model = fsrc("Glm5NextModel", "forward")
import importlib.util
spec = importlib.util.spec_from_file_location("qwmod", f"{d}/glm53_prefill_quickwins.py")
# get the edits without importing torch: parse MHC_FINAL / MHC_AUX tuples
tree = ast.parse(qsrc); edits = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("MHC_FINAL","MHC_AUX"):
        edits[n.targets[0].id] = ast.literal_eval(n.value)
fl, fm = fp(layer), fp(model)
print("live layer fp", fl, "model fp", fm)
print("mhc_aux model ok:", fm in V[("mhc_aux","Glm5NextModel.forward")])
print("mhc_mean model ok:", fm in V[("mhc_mean","Glm5NextModel.forward")])
print("mhc_mean layer ok:", fl in V[("mhc_mean","Glm5NextDecoderLayer.forward")])
o,nw = edits["MHC_FINAL"]; print("MHC_FINAL anchor count", layer.count(o)); 
oa,na = edits["MHC_AUX"]; print("MHC_AUX anchor count", model.count(oa))
lq = textwrap.dedent(layer.replace(o, nw))
print("moeglue warm accepts plain layer:", fl in W["Glm5NextDecoderLayer.forward"], " qw layer:", fp(lq) in W["Glm5NextDecoderLayer.forward"])
print("VERIFIED mhc entries:", {k:sorted(v) for k,v in V.items() if k[0].startswith("mhc")})
print("WARM layer:", sorted(W["Glm5NextDecoderLayer.forward"]))
