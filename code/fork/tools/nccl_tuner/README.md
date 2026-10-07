# NCCL tuner for GLM-5.3 TP=2 prefill all-reduces

`libnccl-tuner-example.so` is NVIDIA's example tuner plugin (NCCL v2.30.7-1, plugins/tuner/example/plugin.c, Apache-2.0),
built in the production image: `gcc -O2 -Inccl -fPIC -shared -o libnccl-tuner-example.so plugin.c`.
`glm53_tuner.conf` sends all-reduces of >= 1 MiB (prefill: 13824-token chunks are 113 MB) to Ring/Simple on the
2-node / 2-rank communicators; decode all-reduces (<= 512 KiB) match no line and keep NCCL's own choice.
Enable on both ranks: NCCL_TUNER_PLUGIN=/opt/glm53/tf/tools/libnccl-tuner-example.so
NCCL_TUNER_CONFIG_FILE=/opt/glm53/tf/tools/glm53_tuner.conf (the bundle dir is mounted at /opt/glm53/tf).
Verified on nodeC: "TUNER/Plugin: Using Example (v6)", "Loaded config: allreduce [1048576-4294967295] ring/simple ... nodes=2 ranks=2".
