// Python bindings for kernels/exl3.cu (docs/DESIGN.md §A). Every TORCH_CHECK in this file runs before the
// first kernel of a call is enqueued, so a check failure never leaves a partial result in `out`.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t tf_s_cap(int64_t P, int64_t n);
int tf_parity();
int tf_num_variants();
int tf_variant();
void tf_set_variant(int);
int tf_sk_table();
int tf_variant_sk_table(int);
bool tf_grouped_launchable(int64_t kdim, int64_t SK, int64_t S_cap);
void launch_route_prep(const at::Tensor&, int64_t, int64_t, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, const at::Tensor&, const at::Tensor&);
void launch_rot_in(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t);
void launch_grouped(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t,
                    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void launch_gateup_epilogue(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double,
                            const at::Tensor* = nullptr, const at::Tensor* = nullptr, int64_t = 0);
void launch_down_epilogue(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                          const at::Tensor* = nullptr, int64_t = 0);
void launch_route_ids(const at::Tensor&, const at::Tensor*, const at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                      int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&);

// gate/up grouped (n tiles a block, warps a block, K splits) and down grouped. TensorFold shipped nt = 8; nt = 4
// (twice the blocks, same K range per warp and split -> bit-identical Z) measured 1.6-5.4% faster for gate/up and
// 1-5% for down on GB10 at T = 1..64 (tests/sweep_grouped_cfg.py, median of 5 rounds, cold weights). A13.
constexpr int64_t GU_NT = 4, GU_W = 4, GU_SK = 4;
constexpr int64_t DN_NT = 4, DN_W = 4, DN_SK = 1;
// Down GEMV K splits for tiny calls: at P <= 8 (T = 1 decode) the down grid has too few blocks to cover the SMs;
// SK = 2 measured 70.3 vs 75.6 us per call (tests/sweep_grouped_cfg.py). Static in P -> CUDA-graph safe.
constexpr int64_t DN_SK_SMALL = 2, DN_SMALL_P = 8;
constexpr int64_t MAX_PAIRS = int64_t(1) << 20;
// Split-count tables (static in P -> CUDA-graph safe), selected by the kernel variant (tf_sk_table(); the shipped
// variant 0 names the production table: table 6, docs/OPTIMIZATION.md K6 - more K splits where the grids are too
// small to cover the SMs (paired A/B of table 1 vs table 0, 0.967-0.999 per call at T = 1..24), and C2 - 2
// gate/up splits above P = 512, where the fp32 partials outgrow the L2 (table 4 vs 1: 0.984-0.991 at T = 96..128,
// 0.994-1.005 at T = 32..64, hence only above P = 512). {gate/up: SK for P <= gu_p, else GU_SK; down: SK dn_sk1 for P <= dn_p1,
// dn_sk2 for P <= dn_p2, else DN_SK}. A wanted SK that does not divide the shape falls back to the largest power
// of two that does (never below the base SK, which the A11 shape check guarantees).
struct SkTable { int64_t gu_p1, gu_sk1, gu_p2, gu_sk2, gu_sk3, dn_p1, dn_sk1, dn_p2, dn_sk2, dn_sk3; };
// gate/up: gu_sk1 for P <= gu_p1, gu_sk2 for P <= gu_p2, else gu_sk3; down likewise
static const SkTable kSkTables[] = {
    {0, GU_SK, 0, GU_SK, GU_SK, DN_SMALL_P, DN_SK_SMALL, 0, DN_SK, DN_SK},   // 0: A13 (gate/up 4; down 2 at P <= 8)
    {32, 8, 0, GU_SK, GU_SK, 192, 2, 0, DN_SK, DN_SK},                       // 1: gate/up 8 at P <= 32; down 2 at P <= 192
    {32, 8, 0, GU_SK, GU_SK, 32, 4, 192, 2, DN_SK},                          // 2: as 1, down 4 at P <= 32
    {8, 8, 0, GU_SK, GU_SK, 192, 2, 0, DN_SK, DN_SK},                        // 3: as 1, gate/up 8 only at P <= 8
    {32, 8, 192, GU_SK, 2, 192, 2, 0, DN_SK, DN_SK},                         // 4: as 1, gate/up 2 at P > 192
    {32, 8, 0, 2, 2, 192, 2, 0, DN_SK, DN_SK},                               // 5: as 1, gate/up 2 at P > 32
    {32, 8, 512, GU_SK, 2, 192, 2, 0, DN_SK, DN_SK},                         // 6: as 1, gate/up 2 at P > 512 (shipped)
};
constexpr int kNumSkTables = sizeof(kSkTables) / sizeof(kSkTables[0]);
static inline int64_t fit_sk(int64_t want, int64_t base, int64_t kdim, int64_t warps) {
    while (want > base && kdim % (16 * want * warps)) want >>= 1;
    return want;
}
static inline int64_t gu_sk_t(int64_t P, int64_t K, int tab) {
    const SkTable& t = kSkTables[tab];
    const int64_t want = P <= t.gu_p1 ? t.gu_sk1 : P <= t.gu_p2 ? t.gu_sk2 : t.gu_sk3;
    return fit_sk(want, GU_SK, K, GU_W);
}
static inline int64_t down_sk_t(int64_t P, int64_t N, int tab) {
    const SkTable& t = kSkTables[tab];
    const int64_t want = P <= t.dn_p1 ? t.dn_sk1 : P <= t.dn_p2 ? t.dn_sk2 : t.dn_sk3;
    return fit_sk(want, DN_SK, N, DN_W);
}
static inline int64_t gu_sk(int64_t P, int64_t K) { return gu_sk_t(P, K, tf_sk_table()); }
static inline int64_t down_sk(int64_t P, int64_t N) { return down_sk_t(P, N, tf_sk_table()); }
// fp32 scratch Z needed by one moe_forward call with P pairs: gate/up partials [2][gu_sk][P][N], then (reusing
// the same buffer) down partials [down_sk][P][K], maximised over every split-count table (so any kernel variant
// fits the same persistent scratch). The single source of truth for both the pre-launch check in moe_forward and
// the persistent scratch size in tf_exl3_moe.py (z_need_max), so a retune can never leave Python sizing Z with
// stale constants.
static inline int64_t z_need(int64_t P, int64_t K, int64_t N) {
    int64_t z = 0;
    for (int tab = 0; tab < kNumSkTables; ++tab)
        z = std::max(z, std::max(2 * gu_sk_t(P, K, tab) * P * N, down_sk_t(P, N, tab) * P * K));
    return z;
}
// max over every P in [1, P_cap] (down_sk makes z_need non-monotone in P, so take the true maximum)
static int64_t z_need_max(int64_t P_cap, int64_t K, int64_t N) {
    TORCH_CHECK(P_cap >= 1 && P_cap <= MAX_PAIRS && K >= 1 && N >= 1, "z_need_max: bad P_cap, K or N");
    int64_t z = 0;
    for (int64_t P = 1; P <= P_cap; ++P) z = std::max(z, z_need(P, K, N));
    return z;
}

static void check(const at::Tensor& x, at::ScalarType t, const char* name, const at::Device& dev) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of dtype ", t, ", got ", x.scalar_type(),
                x.is_contiguous() ? "" : " (non-contiguous)");
    TORCH_CHECK(x.device() == dev, name, ": on ", x.device(), ", expected ", dev);
}

static void check_ptr_table(const at::Tensor& t, int64_t n, const char* name, const at::Device& dev) {
    check(t, at::kLong, name, dev);
    TORCH_CHECK(t.dim() == 1 && t.size(0) >= n, name, ": pointer table must be int64 [>= n]");
}

static void check_min(const at::Tensor& t, int64_t numel, const char* name) {
    TORCH_CHECK(t.numel() >= numel, name, ": scratch too small (", t.numel(), " < ", numel, ")");
}

// ---- stage entry points (unit tests E-U1..E-U4 and debugging) ------------------------------------------

void route_prep(const at::Tensor& expert_count, int64_t P, int64_t R, at::Tensor pair_expert, at::Tensor seg_expert,
                at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor nseg) {
    const auto dev = expert_count.device();
    check(expert_count, at::kLong, "expert_count", dev);
    TORCH_CHECK(expert_count.dim() == 1 && expert_count.size(0) >= 2, "expert_count: int64 [n+1], n >= 1");
    const int64_t n = expert_count.size(0) - 1;
    TORCH_CHECK(P >= 1 && P <= MAX_PAIRS && R >= 1, "route_prep: bad P or R");
    const int64_t S_cap = tf_s_cap(P, n);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");
    c10::cuda::CUDAGuard guard(dev);
    launch_route_prep(expert_count, n, P, R, S_cap, pair_expert, seg_expert, seg_row0, seg_rows, nseg);
}

void rot_in(const at::Tensor& x, const at::Tensor& token_sorted, const at::Tensor& pair_expert,
            const at::Tensor& suh_p0, const at::Tensor& suh_p1, at::Tensor out0, at::Tensor out1) {
    const auto dev = x.device();
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) && x.dim() == 2 &&
                    x.stride(1) == 1 && x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
                "x: fp16/bf16 CUDA [B, K], unit column stride, row stride % 4 == 0, 8-byte aligned");
    const int64_t B = x.size(0), K = x.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    check(token_sorted, at::kLong, "token_sorted", dev);
    const int64_t P = token_sorted.numel();
    TORCH_CHECK(P >= 1 && P <= MAX_PAIRS, "rot_in: bad P");
    check(pair_expert, at::kInt, "pair_expert", dev);
    check_min(pair_expert, P, "pair_expert");
    check(suh_p0, at::kLong, "suh_p0", dev);
    check(suh_p1, at::kLong, "suh_p1", dev);
    check(out0, at::kHalf, "out0", dev);
    check(out1, at::kHalf, "out1", dev);
    check_min(out0, P * K, "out0");
    check_min(out1, P * K, "out1");
    c10::cuda::CUDAGuard guard(dev);
    launch_rot_in(x, x.stride(0), token_sorted, pair_expert, suh_p0, suh_p1, out0, out1, P, K, B);
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& Tp0, const at::Tensor& Tp1,
             const at::Tensor& seg_expert, const at::Tensor& seg_row0, const at::Tensor& seg_rows,
             const at::Tensor& nseg, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t nt, int64_t warps, int64_t S_cap) {
    const auto dev = X0.device();
    check(X0, at::kHalf, "X0", dev);
    check(X1, at::kHalf, "X1", dev);
    check(Tp0, at::kLong, "Tp0", dev);
    check(Tp1, at::kLong, "Tp1", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check(Z, at::kFloat, "Z", dev);
    TORCH_CHECK(mats == 1 || mats == 2, "mats must be 1 or 2");
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    TORCH_CHECK(P >= 1 && P <= MAX_PAIRS && S_cap >= 1, "grouped: bad P or S_cap");
    check_min(X0, P * K, "X0");
    check_min(X1, P * K, "X1");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(Z, mats * SK * P * N, "Z");
    c10::cuda::CUDAGuard guard(dev);
    launch_grouped(X0, X1, Tp0, Tp1, seg_expert, seg_row0, seg_rows, nseg, Z, mats, K, N, P, SK, nt, warps, S_cap);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pair_expert, const at::Tensor& svh_pg,
                     const at::Tensor& svh_pu, const at::Tensor& suh_pd, at::Tensor xd, int64_t P, int64_t N,
                     int64_t SK, double limit) {
    const auto dev = Z.device();
    check(Z, at::kFloat, "Z", dev);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(svh_pg, at::kLong, "svh_pg", dev);
    check(svh_pu, at::kLong, "svh_pu", dev);
    check(suh_pd, at::kLong, "suh_pd", dev);
    check(xd, at::kHalf, "xd", dev);
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(P >= 1 && P <= MAX_PAIRS, "gateup_epilogue: bad P");
    check_min(pair_expert, P, "pair_expert");
    check_min(Z, 2 * SK * P * N, "Z");
    check_min(xd, P * N, "xd");
    c10::cuda::CUDAGuard guard(dev);
    launch_gateup_epilogue(Z, pair_expert, svh_pg, svh_pu, suh_pd, xd, P, N, SK, limit);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pair_expert, const at::Tensor& token_sorted,
                   const at::Tensor& weight_sorted, const at::Tensor& svh_pd, at::Tensor out, int64_t SK) {
    const auto dev = Z.device();
    check(Z, at::kFloat, "Z", dev);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    check(svh_pd, at::kLong, "svh_pd", dev);
    check(out, at::kFloat, "out", dev);
    TORCH_CHECK(out.dim() == 2 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out: fp32 [B, D], 16-byte aligned");
    const int64_t B = out.size(0), D = out.size(1), P = token_sorted.numel();
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    TORCH_CHECK(P >= 1 && P <= MAX_PAIRS && weight_sorted.numel() == P, "down_epilogue: bad P");
    check_min(pair_expert, P, "pair_expert");
    check_min(Z, SK * P * D, "Z");
    c10::cuda::CUDAGuard guard(dev);
    launch_down_epilogue(Z, pair_expert, token_sorted, weight_sorted, svh_pd, out, P, D, SK, B);
}

// ---- the whole exl3_moe-equivalent call: 6 kernels, one stream, static grids ----------------------------

void moe_forward(const at::Tensor& x, at::Tensor out, const at::Tensor& expert_count, const at::Tensor& token_sorted,
                 const at::Tensor& weight_sorted, const at::Tensor& g_t, const at::Tensor& g_suh,
                 const at::Tensor& g_svh, const at::Tensor& u_t, const at::Tensor& u_suh, const at::Tensor& u_svh,
                 const at::Tensor& d_t, const at::Tensor& d_suh, const at::Tensor& d_svh, at::Tensor xg,
                 at::Tensor xu, at::Tensor xd, at::Tensor Z, at::Tensor pair_expert, at::Tensor seg_expert,
                 at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor nseg, int64_t R, int64_t N, double limit) {
    const auto dev = x.device();
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.stride(1) == 1 &&
                    x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x: fp16 CUDA [B, K], unit column stride, row stride % 4 == 0, 16-byte aligned");
    const int64_t B = x.size(0), K = x.size(1);
    TORCH_CHECK(B >= 1, "x: empty batch");
    check(out, at::kFloat, "out", dev);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == K &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "out: fp32 [B, K], 16-byte aligned");
    check(expert_count, at::kLong, "expert_count", dev);
    TORCH_CHECK(expert_count.dim() == 1 && expert_count.size(0) >= 2, "expert_count: int64 [n+1], n >= 1");
    const int64_t n = expert_count.size(0) - 1;
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    const int64_t P = token_sorted.numel();
    TORCH_CHECK(token_sorted.dim() == 1 && weight_sorted.dim() == 1 && weight_sorted.numel() == P && P >= 1 &&
                    P <= MAX_PAIRS,
                "token_sorted / weight_sorted: 1-D, same length, 1 <= P <= 2^20");
    check_ptr_table(g_t, n, "gate_ptrs_trellis", dev);
    check_ptr_table(g_suh, n, "gate_ptrs_suh", dev);
    check_ptr_table(g_svh, n, "gate_ptrs_svh", dev);
    check_ptr_table(u_t, n, "up_ptrs_trellis", dev);
    check_ptr_table(u_suh, n, "up_ptrs_suh", dev);
    check_ptr_table(u_svh, n, "up_ptrs_svh", dev);
    check_ptr_table(d_t, n, "down_ptrs_trellis", dev);
    check_ptr_table(d_suh, n, "down_ptrs_suh", dev);
    check_ptr_table(d_svh, n, "down_ptrs_svh", dev);
    // A11: grouped splits (gate/up K % (16*SK*W), N % (16*nt); down N % (16*SK*W), K % (16*nt)) and the
    // 128-wide Hadamard blocks of rot_in / gateup_epilogue / down_epilogue
    TORCH_CHECK(K % (16 * GU_SK * GU_W) == 0 && N % (16 * GU_NT) == 0 && N % (16 * DN_SK * DN_W) == 0 &&
                    K % (16 * DN_NT) == 0 && K % 256 == 0 && N % 128 == 0,
                "shape: need hidden % 256 == 0 and intermediate % 128 == 0 (A11)");
    TORCH_CHECK(R >= 1, "R must be >= 1");
    const int64_t S_cap = tf_s_cap(P, n);
    check(xg, at::kHalf, "xg", dev);
    check(xu, at::kHalf, "xu", dev);
    check(xd, at::kHalf, "xd", dev);
    check(Z, at::kFloat, "Z", dev);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check_min(xg, P * K, "xg");
    check_min(xu, P * K, "xu");
    check_min(xd, P * N, "xd");
    const int64_t dn_sk = down_sk(P, N), g_sk = gu_sk(P, K);
    TORCH_CHECK(N % (16 * dn_sk * DN_W) == 0 && K % (16 * g_sk * GU_W) == 0,
                "shape: hidden / intermediate must split over the K splits");
    // the grouped launches' variant resolution (incl. the ORD 0 fallback at S_cap > 65535) must find a compiled
    // instance; checked here so that launch_grouped cannot throw after the first kernel is enqueued
    TORCH_CHECK(tf_grouped_launchable(K, g_sk, S_cap) && tf_grouped_launchable(N, dn_sk, S_cap),
                "grouped kernel variant not instantiated for this launch");
    check_min(Z, z_need(P, K, N), "Z");
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");

    c10::cuda::CUDAGuard guard(dev);
    launch_route_prep(expert_count, n, P, R, S_cap, pair_expert, seg_expert, seg_row0, seg_rows, nseg);
    launch_rot_in(x, x.stride(0), token_sorted, pair_expert, g_suh, u_suh, xg, xu, P, K, B);
    launch_grouped(xg, xu, g_t, u_t, seg_expert, seg_row0, seg_rows, nseg, Z, 2, K, N, P, g_sk, GU_NT, GU_W, S_cap);
    launch_gateup_epilogue(Z, pair_expert, g_svh, u_svh, d_suh, xd, P, N, g_sk, limit, &xg, &xu, K);
    launch_grouped(xd, xd, d_t, d_t, seg_expert, seg_row0, seg_rows, nseg, Z, 1, N, K, P, dn_sk, DN_NT, DN_W, S_cap);
    launch_down_epilogue(Z, pair_expert, token_sorted, weight_sorted, d_svh, out, P, K, dn_sk, B, &xd, N);
}

// ---- K2: the whole decode apply after routing selection, from the router's ids (docs/OPTIMIZATION.md) ------
// Replaces production's decode prelude (map_topk_to_local, argsort, gathers, expert_count, x2d.half()) plus the
// exl3_moe call: route_ids -> rot_in (x read as bf16 or fp16) -> grouped(g,u) -> gateup_epilogue -> grouped(d)
// -> down_epilogue, accumulating into `out`, which the caller zeroes (production allocates it with torch.zeros).
// Every check below runs before the first launch.
constexpr int64_t ROUTE_IDS_MAX_PAIRS = 1024, ROUTE_IDS_MAX_EXPERTS = 4096;

void moe_forward_ids(const at::Tensor& x, at::Tensor out, const at::Tensor& ids, const at::Tensor& weights,
                     const c10::optional<at::Tensor>& expert_map, int64_t n, const at::Tensor& g_t,
                     const at::Tensor& g_suh, const at::Tensor& g_svh, const at::Tensor& u_t, const at::Tensor& u_suh,
                     const at::Tensor& u_svh, const at::Tensor& d_t, const at::Tensor& d_suh, const at::Tensor& d_svh,
                     at::Tensor xg, at::Tensor xu, at::Tensor xd, at::Tensor Z, at::Tensor pair_expert,
                     at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor nseg,
                     at::Tensor token_sorted, at::Tensor weight_sorted, int64_t R, int64_t N, double limit) {
    const auto dev = x.device();
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.dim() == 2 &&
                    x.stride(1) == 1 && x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
                "x: bf16/fp16 CUDA [B, K], unit column stride, row stride % 4 == 0, 8-byte aligned");
    const int64_t B = x.size(0), K = x.size(1);
    TORCH_CHECK(B >= 1, "x: empty batch");
    check(out, at::kFloat, "out", dev);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == B && out.size(1) == K &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "out: fp32 [B, K], 16-byte aligned");
    check(ids, at::kLong, "ids", dev);
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) == B && ids.size(1) >= 1, "ids: int64 [B, topk]");
    const int64_t topk = ids.size(1), P = B * topk;
    TORCH_CHECK(P <= ROUTE_IDS_MAX_PAIRS, "moe_forward_ids: B * topk must be <= ", ROUTE_IDS_MAX_PAIRS);
    TORCH_CHECK(weights.is_cuda() && weights.device() == dev && weights.is_contiguous() &&
                    (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kBFloat16 ||
                     weights.scalar_type() == at::kHalf) && weights.sizes() == ids.sizes(),
                "weights: contiguous fp32/bf16/fp16 CUDA tensor shaped like ids");
    const at::Tensor* em = nullptr;
    if (expert_map.has_value()) {
        check(*expert_map, at::kLong, "expert_map", dev);
        TORCH_CHECK(expert_map->dim() == 1, "expert_map: int64 [n_global]");
        em = &*expert_map;
    }
    TORCH_CHECK(n >= 1 && n <= ROUTE_IDS_MAX_EXPERTS, "moe_forward_ids: 1 <= n <= ", ROUTE_IDS_MAX_EXPERTS);
    check_ptr_table(g_t, n, "gate_ptrs_trellis", dev);
    check_ptr_table(g_suh, n, "gate_ptrs_suh", dev);
    check_ptr_table(g_svh, n, "gate_ptrs_svh", dev);
    check_ptr_table(u_t, n, "up_ptrs_trellis", dev);
    check_ptr_table(u_suh, n, "up_ptrs_suh", dev);
    check_ptr_table(u_svh, n, "up_ptrs_svh", dev);
    check_ptr_table(d_t, n, "down_ptrs_trellis", dev);
    check_ptr_table(d_suh, n, "down_ptrs_suh", dev);
    check_ptr_table(d_svh, n, "down_ptrs_svh", dev);
    TORCH_CHECK(K % (16 * GU_SK * GU_W) == 0 && N % (16 * GU_NT) == 0 && N % (16 * DN_SK * DN_W) == 0 &&
                    K % (16 * DN_NT) == 0 && K % 256 == 0 && N % 128 == 0,
                "shape: need hidden % 256 == 0 and intermediate % 128 == 0 (A11)");
    TORCH_CHECK(R >= 1, "R must be >= 1");
    const int64_t S_cap = tf_s_cap(P, n);
    check(xg, at::kHalf, "xg", dev);
    check(xu, at::kHalf, "xu", dev);
    check(xd, at::kHalf, "xd", dev);
    check(Z, at::kFloat, "Z", dev);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    check_min(xg, P * K, "xg");
    check_min(xu, P * K, "xu");
    check_min(xd, P * N, "xd");
    const int64_t dn_sk = down_sk(P, N), g_sk = gu_sk(P, K);
    TORCH_CHECK(N % (16 * dn_sk * DN_W) == 0 && K % (16 * g_sk * GU_W) == 0,
                "shape: hidden / intermediate must split over the K splits");
    // the grouped launches' variant resolution (incl. the ORD 0 fallback at S_cap > 65535) must find a compiled
    // instance; checked here so that launch_grouped cannot throw after the first kernel is enqueued
    TORCH_CHECK(tf_grouped_launchable(K, g_sk, S_cap) && tf_grouped_launchable(N, dn_sk, S_cap),
                "grouped kernel variant not instantiated for this launch");
    check_min(Z, z_need(P, K, N), "Z");
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");
    check_min(token_sorted, P, "token_sorted");
    check_min(weight_sorted, P, "weight_sorted");

    c10::cuda::CUDAGuard guard(dev);
    launch_route_ids(ids, em, weights, P, topk, n, R, S_cap, pair_expert, seg_expert, seg_row0, seg_rows, nseg,
                     token_sorted, weight_sorted);
    launch_rot_in(x, x.stride(0), token_sorted, pair_expert, g_suh, u_suh, xg, xu, P, K, B);
    launch_grouped(xg, xu, g_t, u_t, seg_expert, seg_row0, seg_rows, nseg, Z, 2, K, N, P, g_sk, GU_NT, GU_W, S_cap);
    launch_gateup_epilogue(Z, pair_expert, g_svh, u_svh, d_suh, xd, P, N, g_sk, limit, &xg, &xu, K);
    launch_grouped(xd, xd, d_t, d_t, seg_expert, seg_row0, seg_rows, nseg, Z, 1, N, K, P, dn_sk, DN_NT, DN_W, S_cap);
    launch_down_epilogue(Z, pair_expert, token_sorted, weight_sorted, d_svh, out, P, K, dn_sk, B, &xd, N);
}

// ---- moeglue (docs/DEC_MOEGLUE.md, GLM53_DEC_MOEGLUE): production's whole decode apply_exl3_experts ------------
// From the router's raw topk ids (int32 as the router returns them, or int64) to the routed output in x's dtype:
// glue_prep -> grouped(g,u) -> gateup_epilogue -> grouped(d) -> glue_finish. Replaces the ids .to(long) copy,
// torch.zeros(out), route_ids, rot_in, down_epilogue and out.to(x.dtype). `out` is written, not accumulated (no
// zeroing needed). Every check below runs before the first launch.
int64_t tf_glue_prep_blocks(int64_t P, int64_t K);
void launch_glue_prep(const at::Tensor&, const at::Tensor&, const at::Tensor*, const at::Tensor&, int64_t, int64_t,
                      int64_t, int64_t, int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, bool, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
void launch_glue_finish(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                        const at::Tensor*, int64_t);
constexpr int64_t GLUE_MAX_TOPK = 32, GLUE_MAX_PAIRS = 4096;

void moe_forward_glue(const at::Tensor& x, at::Tensor out, const at::Tensor& ids, const at::Tensor& weights,
                      const c10::optional<at::Tensor>& expert_map, int64_t n, const at::Tensor& g_t,
                      const at::Tensor& g_suh, const at::Tensor& g_svh, const at::Tensor& u_t, const at::Tensor& u_suh,
                      const at::Tensor& u_svh, const at::Tensor& d_t, const at::Tensor& d_suh, const at::Tensor& d_svh,
                      at::Tensor xg, at::Tensor xu, at::Tensor xd, at::Tensor Z, at::Tensor pair_expert,
                      at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor nseg,
                      at::Tensor token_sorted, at::Tensor weight_sorted, at::Tensor inv, int64_t R, int64_t N,
                      double limit, bool prefetch, int64_t parts) {
    const auto dev = x.device();
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.dim() == 2 &&
                    x.stride(1) == 1 && x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
                "x: bf16/fp16 CUDA [B, K], unit column stride, row stride % 4 == 0, 8-byte aligned");
    const int64_t B = x.size(0), K = x.size(1);
    TORCH_CHECK(B >= 1, "x: empty batch");
    TORCH_CHECK(out.is_cuda() && out.device() == dev && out.scalar_type() == x.scalar_type() && out.dim() == 2 &&
                    out.size(0) == B && out.size(1) == K && out.stride(1) == 1 && out.stride(0) >= K,
                "out: x's dtype, [B, K], unit column stride");
    TORCH_CHECK(ids.is_cuda() && ids.device() == dev && ids.is_contiguous() &&
                    (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong) && ids.dim() == 2 &&
                    ids.size(0) == B && ids.size(1) >= 1,
                "ids: contiguous int32/int64 CUDA [B, topk]");
    const int64_t topk = ids.size(1), P = B * topk;
    TORCH_CHECK(topk <= GLUE_MAX_TOPK && P <= GLUE_MAX_PAIRS, "moe_forward_glue: topk <= ", GLUE_MAX_TOPK,
                " and B * topk <= ", GLUE_MAX_PAIRS);
    TORCH_CHECK(weights.is_cuda() && weights.device() == dev && weights.is_contiguous() &&
                    (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kBFloat16 ||
                     weights.scalar_type() == at::kHalf) && weights.sizes() == ids.sizes(),
                "weights: contiguous fp32/bf16/fp16 CUDA tensor shaped like ids");
    const at::Tensor* em = nullptr;
    if (expert_map.has_value()) {
        check(*expert_map, at::kLong, "expert_map", dev);
        TORCH_CHECK(expert_map->dim() == 1, "expert_map: int64 [n_global]");
        em = &*expert_map;
    }
    TORCH_CHECK(n >= 1 && n <= ROUTE_IDS_MAX_EXPERTS, "moe_forward_glue: 1 <= n <= ", ROUTE_IDS_MAX_EXPERTS);
    check_ptr_table(g_t, n, "gate_ptrs_trellis", dev);
    check_ptr_table(g_suh, n, "gate_ptrs_suh", dev);
    check_ptr_table(g_svh, n, "gate_ptrs_svh", dev);
    check_ptr_table(u_t, n, "up_ptrs_trellis", dev);
    check_ptr_table(u_suh, n, "up_ptrs_suh", dev);
    check_ptr_table(u_svh, n, "up_ptrs_svh", dev);
    check_ptr_table(d_t, n, "down_ptrs_trellis", dev);
    check_ptr_table(d_suh, n, "down_ptrs_suh", dev);
    check_ptr_table(d_svh, n, "down_ptrs_svh", dev);
    TORCH_CHECK(K % (16 * GU_SK * GU_W) == 0 && N % (16 * GU_NT) == 0 && N % (16 * DN_SK * DN_W) == 0 &&
                    K % (16 * DN_NT) == 0 && K % 512 == 0 && N % 128 == 0,
                "shape: need hidden % 512 == 0 and intermediate % 128 == 0 (A11, glue_prep column groups)");
    TORCH_CHECK(R >= 1, "R must be >= 1");
    const int64_t S_cap = tf_s_cap(P, n);
    check(xg, at::kHalf, "xg", dev);
    check(xu, at::kHalf, "xu", dev);
    check(xd, at::kHalf, "xd", dev);
    check(Z, at::kFloat, "Z", dev);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    check(inv, at::kInt, "inv", dev);
    check_min(xg, P * K, "xg");
    check_min(xu, P * K, "xu");
    check_min(xd, P * N, "xd");
    const int64_t dn_sk = down_sk(P, N), g_sk = gu_sk(P, K);
    TORCH_CHECK(N % (16 * dn_sk * DN_W) == 0 && K % (16 * g_sk * GU_W) == 0,
                "shape: hidden / intermediate must split over the K splits");
    TORCH_CHECK(tf_grouped_launchable(K, g_sk, S_cap) && tf_grouped_launchable(N, dn_sk, S_cap),
                "grouped kernel variant not instantiated for this launch");
    TORCH_CHECK(tf_glue_prep_blocks(P, K) <= 0x7fffffffLL, "glue_prep grid too large");
    check_min(Z, z_need(P, K, N), "Z");
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");
    check_min(token_sorted, P, "token_sorted");
    check_min(weight_sorted, P, "weight_sorted");
    check_min(inv, P, "inv");

    TORCH_CHECK(parts >= 1 && parts <= 3, "parts: 1 = glue_prep + grouped g/u, 2 = the rest, 3 = all");
    c10::cuda::CUDAGuard guard(dev);
    if (parts & 1) {
        launch_glue_prep(x, ids, em, weights, P, topk, n, R, S_cap, K, g_suh, u_suh, g_svh, u_svh, d_suh, d_svh, N,
                         prefetch, xg, xu, pair_expert, seg_expert, seg_row0, seg_rows, nseg, token_sorted,
                         weight_sorted, inv);
        launch_grouped(xg, xu, g_t, u_t, seg_expert, seg_row0, seg_rows, nseg, Z, 2, K, N, P, g_sk, GU_NT, GU_W, S_cap);
    }
    if (parts & 2) {
        launch_gateup_epilogue(Z, pair_expert, g_svh, u_svh, d_suh, xd, P, N, g_sk, limit, &xg, &xu, K);
        launch_grouped(xd, xd, d_t, d_t, seg_expert, seg_row0, seg_rows, nseg, Z, 1, N, K, P, dn_sk, DN_NT, DN_W,
                       S_cap);
        launch_glue_finish(Z, pair_expert, inv, weight_sorted, d_svh, out, P, K, dn_sk, topk, B, &xd, N);
    }
}

// stage entry for tests: glue_prep alone (routing tables + the rotated gate/up inputs)
void glue_prep(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& weights,
               const c10::optional<at::Tensor>& expert_map, int64_t n, int64_t R, const at::Tensor& g_suh,
               const at::Tensor& u_suh, at::Tensor xg, at::Tensor xu, at::Tensor pair_expert, at::Tensor seg_expert,
               at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor nseg, at::Tensor token_sorted,
               at::Tensor weight_sorted, at::Tensor inv, const c10::optional<at::Tensor>& g_svh,
               const c10::optional<at::Tensor>& u_svh, const c10::optional<at::Tensor>& d_suh,
               const c10::optional<at::Tensor>& d_svh, int64_t N, bool prefetch) {
    const auto dev = x.device();
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.dim() == 2 &&
                    x.stride(1) == 1 && x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
                "x: bf16/fp16 CUDA [B, K]");
    const int64_t B = x.size(0), K = x.size(1);
    TORCH_CHECK(K % 512 == 0, "K % 512");
    TORCH_CHECK(ids.is_cuda() && ids.device() == dev && ids.is_contiguous() &&
                    (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong) && ids.dim() == 2 &&
                    ids.size(0) == B && ids.size(1) >= 1 && ids.size(1) <= GLUE_MAX_TOPK,
                "ids: contiguous int32/int64 CUDA [B, topk]");
    const int64_t topk = ids.size(1), P = B * topk;
    TORCH_CHECK(P <= GLUE_MAX_PAIRS && n >= 1 && n <= ROUTE_IDS_MAX_EXPERTS && R >= 1, "glue_prep: bad P, n or R");
    TORCH_CHECK(weights.is_cuda() && weights.device() == dev && weights.is_contiguous() &&
                    (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kBFloat16 ||
                     weights.scalar_type() == at::kHalf) && weights.sizes() == ids.sizes(),
                "weights: shaped like ids");
    const at::Tensor* em = nullptr;
    if (expert_map.has_value()) {
        check(*expert_map, at::kLong, "expert_map", dev);
        em = &*expert_map;
    }
    check_ptr_table(g_suh, n, "gate_ptrs_suh", dev);
    check_ptr_table(u_suh, n, "up_ptrs_suh", dev);
    const int64_t S_cap = tf_s_cap(P, n);
    check(xg, at::kHalf, "xg", dev);
    check(xu, at::kHalf, "xu", dev);
    check_min(xg, P * K, "xg");
    check_min(xu, P * K, "xu");
    for (const at::Tensor* t : {&pair_expert, &seg_expert, &seg_row0, &seg_rows, &nseg, &inv}) check(*t, at::kInt, "i32", dev);
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");
    check_min(token_sorted, P, "token_sorted");
    check_min(weight_sorted, P, "weight_sorted");
    check_min(inv, P, "inv");
    const bool pf = prefetch && g_svh.has_value() && u_svh.has_value() && d_suh.has_value() && d_svh.has_value();
    if (pf) {
        check_ptr_table(*g_svh, n, "gate_ptrs_svh", dev);
        check_ptr_table(*u_svh, n, "up_ptrs_svh", dev);
        check_ptr_table(*d_suh, n, "down_ptrs_suh", dev);
        check_ptr_table(*d_svh, n, "down_ptrs_svh", dev);
        TORCH_CHECK(N >= 64 && N % 64 == 0, "N");
    }
    c10::cuda::CUDAGuard guard(dev);
    launch_glue_prep(x, ids, em, weights, P, topk, n, R, S_cap, K, g_suh, u_suh, pf ? *g_svh : g_suh,
                     pf ? *u_svh : u_suh, pf ? *d_suh : g_suh, pf ? *d_svh : g_suh, pf ? N : 128, pf, xg, xu,
                     pair_expert, seg_expert, seg_row0, seg_rows, nseg, token_sorted, weight_sorted, inv);
}

// stage entry for tests: route_ids alone (production routing + route_prep)
void route_ids(const at::Tensor& ids, const at::Tensor& weights, const c10::optional<at::Tensor>& expert_map,
               int64_t n, int64_t R, at::Tensor pair_expert, at::Tensor seg_expert, at::Tensor seg_row0,
               at::Tensor seg_rows, at::Tensor nseg, at::Tensor token_sorted, at::Tensor weight_sorted) {
    const auto dev = ids.device();
    check(ids, at::kLong, "ids", dev);
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) >= 1 && ids.size(1) >= 1, "ids: int64 [B, topk]");
    const int64_t topk = ids.size(1), P = ids.size(0) * topk;
    TORCH_CHECK(P <= ROUTE_IDS_MAX_PAIRS && n >= 1 && n <= ROUTE_IDS_MAX_EXPERTS && R >= 1, "route_ids: bad P, n or R");
    TORCH_CHECK(weights.is_cuda() && weights.device() == dev && weights.is_contiguous() &&
                    (weights.scalar_type() == at::kFloat || weights.scalar_type() == at::kBFloat16 ||
                     weights.scalar_type() == at::kHalf) && weights.sizes() == ids.sizes(),
                "weights: contiguous fp32/bf16/fp16 CUDA tensor shaped like ids");
    const at::Tensor* em = nullptr;
    if (expert_map.has_value()) {
        check(*expert_map, at::kLong, "expert_map", dev);
        TORCH_CHECK(expert_map->dim() == 1, "expert_map: int64 [n_global]");
        em = &*expert_map;
    }
    const int64_t S_cap = tf_s_cap(P, n);
    check(pair_expert, at::kInt, "pair_expert", dev);
    check(seg_expert, at::kInt, "seg_expert", dev);
    check(seg_row0, at::kInt, "seg_row0", dev);
    check(seg_rows, at::kInt, "seg_rows", dev);
    check(nseg, at::kInt, "nseg", dev);
    check(token_sorted, at::kLong, "token_sorted", dev);
    check(weight_sorted, at::kHalf, "weight_sorted", dev);
    check_min(pair_expert, P, "pair_expert");
    check_min(seg_expert, S_cap, "seg_expert");
    check_min(seg_row0, S_cap, "seg_row0");
    check_min(seg_rows, S_cap, "seg_rows");
    check_min(nseg, 1, "nseg");
    check_min(token_sorted, P, "token_sorted");
    check_min(weight_sorted, P, "weight_sorted");
    c10::cuda::CUDAGuard guard(dev);
    launch_route_ids(ids, em, weights, P, topk, n, R, S_cap, pair_expert, seg_expert, seg_row0, seg_rows, nseg,
                     token_sorted, weight_sorted);
}

// moeglue warm (GLM53_DEC_MOEGLUE_WARM): read `regions` (whole tensors, in order) into the L2 on the current stream.
int64_t tf_warm_max_regions();
void launch_l2_warm(const std::vector<const void*>&, const std::vector<int64_t>&, int64_t, int64_t, unsigned*);

void l2_warm(const std::vector<at::Tensor>& regions, int64_t blocks, int64_t unroll, at::Tensor sink) {
    TORCH_CHECK(!regions.empty() && (int64_t)regions.size() <= tf_warm_max_regions(), "l2_warm: 1..",
                tf_warm_max_regions(), " regions");
    TORCH_CHECK(blocks >= 1 && blocks <= 1024, "l2_warm: blocks in 1..1024");
    TORCH_CHECK(unroll == 1 || unroll == 4 || unroll == 8 || unroll == -1 || unroll == -4 || unroll == -8,
                "l2_warm: unroll 1, 4 or 8 (negative: L1::no_allocate loads, tests)");
    TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kInt && sink.numel() >= 1, "l2_warm: sink int32 CUDA");
    const auto dev = sink.device();
    std::vector<const void*> ptrs;
    std::vector<int64_t> bytes;
    for (const auto& t : regions) {
        TORCH_CHECK(t.is_cuda() && t.device() == dev, "l2_warm: every region on the sink's device");
        TORCH_CHECK(t.is_contiguous(), "l2_warm: regions must be contiguous");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "l2_warm: regions must be 16 B aligned");
        ptrs.push_back(t.data_ptr());
        bytes.push_back((t.numel() * (int64_t)t.element_size()) / 16 * 16);
    }
    c10::cuda::CUDAGuard guard(dev);
    launch_l2_warm(ptrs, bytes, blocks, unroll, reinterpret_cast<unsigned*>(sink.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TensorFold EXL3 routed-expert kernels as a drop-in for exllamav3_ext.exl3_moe (docs/DESIGN.md)";
    m.def("moe_forward", &moe_forward, "route_prep -> rot_in -> grouped(g,u) -> gateup_epilogue -> grouped(d) -> down_epilogue");
    m.def("route_prep", &route_prep);
    m.def("moe_forward_ids", &moe_forward_ids,
          "K2: route_ids -> rot_in -> grouped(g,u) -> gateup_epilogue -> grouped(d) -> down_epilogue from router ids");
    m.def("route_ids", &route_ids);
    m.def("moe_forward_glue", &moe_forward_glue,
          "moeglue: glue_prep -> grouped(g,u) -> gateup_epilogue -> grouped(d) -> glue_finish from raw router ids");
    m.def("glue_prep", &glue_prep);
    m.def("l2_warm", &l2_warm, "moeglue warm: read whole tensors (in order) into the L2 on the current stream");
    m.attr("GLUE_LIMITS") = py::make_tuple(GLUE_MAX_TOPK, GLUE_MAX_PAIRS);
    m.attr("ROUTE_IDS_LIMITS") = py::make_tuple(ROUTE_IDS_MAX_PAIRS, ROUTE_IDS_MAX_EXPERTS);
    m.def("rot_in", &rot_in);
    m.def("grouped", &grouped);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
    m.def("s_cap", &tf_s_cap);
    m.def("parity", &tf_parity);
    // test/bench hook (docs/OPTIMIZATION.md): grouped-GEMV kernel variant for later launches (0 = shipped, 1 = the
    // master kernel, 2.. experiments); read on the host at launch time, so a captured graph keeps its variant
    m.def("set_variant", [](int64_t v) { tf_set_variant((int)v); });
    m.def("variant", &tf_variant);
    m.def("num_variants", &tf_num_variants);
    m.def("split_counts", [](int64_t P, int64_t K, int64_t N) { return py::make_tuple(gu_sk(P, K), down_sk(P, N)); },
          "(gate/up K splits, down K splits) moe_forward uses for P pairs with the current kernel variant");
    m.def("variant_sk_table", [](int64_t v) {
        TORCH_CHECK(v >= 0 && v < tf_num_variants(), "bad variant");
        return tf_variant_sk_table((int)v);
    });
    m.def("z_need", [](int64_t P, int64_t K, int64_t N) { return z_need(P, K, N); },
          "fp32 scratch elements one moe_forward call with P pairs needs (the pre-launch check's bound)");
    m.def("z_need_max", &z_need_max, "max of z_need(P, K, N) over P in [1, P_cap] (persistent scratch size)");
    m.attr("GATEUP_CFG") = py::make_tuple(GU_NT, GU_W, GU_SK);
    m.attr("DOWN_CFG") = py::make_tuple(DN_NT, DN_W, DN_SK);
    m.attr("DOWN_SMALL") = py::make_tuple(DN_SMALL_P, DN_SK_SMALL);
}
