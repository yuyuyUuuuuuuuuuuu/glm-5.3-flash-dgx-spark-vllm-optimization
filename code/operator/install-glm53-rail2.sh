#!/bin/bash
# v2. Run once per node as root:  sudo bash ~/tf-exl3-deploy/install-glm53-rail2.sh
# Installs /usr/local/sbin/glm53-rail2 (v2), keeps the NOPASSWD rule for exactly that command, replaces the v1 netplan
# file with glm53-rail2.service (runs "glm53-rail2 boot" before docker at every boot: only the used second-half netdev
# comes up, with trimmed RX rings), and takes the unused second-half netdev down now (nothing uses it). The rail NCCL
# uses (enP2p1s0f0np0) is not touched now; its rings are trimmed at the next GLM stop or boot.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
U="${SUDO_USER:?run via sudo from your normal account}"
SRC="$(dirname "$(readlink -f "$0")")/glm53-rail2"
install -o root -g root -m 0755 "$SRC" /usr/local/sbin/glm53-rail2
T=$(mktemp); echo "$U ALL=(root) NOPASSWD: /usr/local/sbin/glm53-rail2" > "$T"; visudo -cf "$T" >/dev/null
install -o root -g root -m 0440 "$T" /etc/sudoers.d/glm53-rail2; rm -f "$T"
rm -f /etc/netplan/99-cx7-p2.yaml
cat > /etc/systemd/system/glm53-rail2.service <<'UNIT'
[Unit]
Description=GLM-5.3 TP=2 second PCIe-half RoCE rail (glm53-rail2 boot)
After=systemd-networkd.service network.target
Wants=systemd-networkd.service
Before=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/glm53-rail2 boot

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable glm53-rail2.service >/dev/null 2>&1
/usr/local/sbin/glm53-rail2 trim >/dev/null
echo "installed glm53-rail2 v2 + glm53-rail2.service (enabled) for $U on $(hostname); v1 netplan file removed"
/usr/local/sbin/glm53-rail2 status
