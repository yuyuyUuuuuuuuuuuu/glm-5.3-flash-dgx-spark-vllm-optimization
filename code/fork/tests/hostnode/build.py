"""Build tests/hostnode/cb.so (the C host function the host-node probes/tests capture) if it is missing or stale."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, LIB = os.path.join(HERE, "cb.c"), os.path.join(HERE, "cb.so")


def ensure() -> str:
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        out = LIB if os.access(HERE, os.W_OK) else "/tmp/glm53_hostnode_cb.so"
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", out, SRC])
        return out
    return LIB


if __name__ == "__main__":
    print(ensure())
