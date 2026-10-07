#!/usr/bin/env bash
# Detect the RoCE v2 GID index that carries each CX7 device's own IPv4 address (it moved 3<->4 across reboots) and make
# .env's HEAD_GID / WORKER_GID match before start. HEAD_CX7_IB / WORKER_CX7_IB may be comma lists (dual rail): every
# device of a rank must resolve to the SAME index (NCCL takes one NCCL_IB_GID_INDEX per process), else this fails.
# Run on nodeA (head). Prints what it did; exits non-zero if undetectable.
set -uo pipefail
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
val(){ grep -E "^$1=" .env | head -1 | cut -d= -f2 | awk '{print $1}'; }
detect='for dev in $(echo "$1" | tr , " "); do D=/sys/class/infiniband/$dev/ports/1
  nd=$(ls /sys/class/infiniband/$dev/device/net 2>/dev/null | head -1)
  ip=$(ip -4 -o addr show dev "$nd" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -1)
  [ -n "$ip" ] || { echo "no-ipv4:$dev" >&2; exit 1; }
  hex=$(printf "%02x%02x:%02x%02x" $(echo $ip | tr . " ")); f=""
  for i in $(ls $D/gids); do g=$(cat $D/gids/$i 2>/dev/null); t=$(cat $D/gid_attrs/types/$i 2>/dev/null)
    case "$g" in *":ffff:$hex") [ "$t" = "RoCE v2" ] && { f=$i; break; };; esac; done
  [ -n "$f" ] || { echo "no-gid:$dev" >&2; exit 1; }
  echo $f; done | sort -u | { read -r a; read -r b && { echo "mismatch:$a,$b" >&2; exit 1; }; echo $a; }'
hdev=$(val HEAD_CX7_IB); wdev=$(val WORKER_CX7_IB); wip=$(val WORKER_IP); wu=$(val WORKER_USER)
hg=$(bash -c "$detect" _ "$hdev") || { echo "fix_gids: cannot detect one head gid for ($hdev)"; exit 1; }
wg=$(ssh -o BatchMode=yes "$wu@$wip" "bash -c '$detect' _ $wdev") || { echo "fix_gids: cannot detect one worker gid for ($wdev)"; exit 1; }
ch=0
for pair in "HEAD_GID:$hg" "WORKER_GID:$wg"; do k=${pair%%:*}; v=${pair#*:}; cur=$(val $k)
  if [ "$cur" != "$v" ]; then [ $ch = 0 ] && cp -p .env .env.bak-gid-$(date +%Y%m%d-%H%M%S); ch=1
    sed -i -E "s/^$k=[0-9]+(.*)$/$k=$v  # auto (fix_gids $(date +%F\ %T)): RoCE v2 IPv4 gid; was $cur/" .env; echo "fix_gids: $k $cur -> $v"
  else echo "fix_gids: $k=$cur ok ($([ $k = HEAD_GID ] && echo $hdev || echo $wdev))"; fi
done
