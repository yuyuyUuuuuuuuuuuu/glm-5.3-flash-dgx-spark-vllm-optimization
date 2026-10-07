#!/bin/bash
# Run once per node as root:  sudo bash ~/tf-exl3-deploy/install-glm53-rail2.sh
# Installs /usr/local/sbin/glm53-rail2 and a NOPASSWD rule for exactly that command (for the invoking user),
# and writes the boot-time netplan file. It does NOT bring the new links up now: that is done later while GLM
# is stopped, so the live TP=2 link is never touched during serving.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
U="${SUDO_USER:?run via sudo from your normal account}"
SRC="$(dirname "$(readlink -f "$0")")/glm53-rail2"
install -o root -g root -m 0755 "$SRC" /usr/local/sbin/glm53-rail2
F=/etc/sudoers.d/glm53-rail2
T=$(mktemp)
echo "$U ALL=(root) NOPASSWD: /usr/local/sbin/glm53-rail2" > "$T"
visudo -cf "$T" >/dev/null
install -o root -g root -m 0440 "$T" "$F"; rm -f "$T"
/usr/local/sbin/glm53-rail2 persist
echo "installed /usr/local/sbin/glm53-rail2 and $F for $U on $(hostname)"
/usr/local/sbin/glm53-rail2 status
