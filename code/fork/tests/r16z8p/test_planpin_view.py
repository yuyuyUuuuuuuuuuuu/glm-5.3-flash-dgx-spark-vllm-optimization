"""r16z8p helper: the kernel-visible sparse-MLA plan state of an _SM90State (see test_planpin.ws_view)."""
import torch

NUM_SM = None


def bufs_of(s):
    w = s.wrapper
    return [w._qo_indptr_buf, w._kv_indptr_buf, w._kv_len_arr_buf, w._int_workspace_buffer]


def ws_view(ws_bytes, info):
    """flashinfer MLAPlan (scheduler.cuh): work arrays hold total_num_works written entries (allocated 16384, the tail
    is stale and never read), the five merge arrays num_sm entries, work_indptr num_clusters + 1."""
    global NUM_SM
    if NUM_SM is None:
        NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count
    info = [int(x) for x in info]
    w = ws_bytes.view(torch.int32)
    ncl = info[1]
    wi = w[info[15] // 4: info[15] // 4 + ncl + 1]
    tot = int(wi[-1])
    parts = [wi]
    for k in (2, 3, 4, 10, 11, 12, 13, 14):
        parts.append(w[info[k] // 4: info[k] // 4 + tot])
    for k in (5, 6, 7, 8, 9):
        parts.append(w[info[k] // 4: info[k] // 4 + NUM_SM])
    return torch.cat(parts)


def same_view(b1, i1, b2, i2):
    if list(i1) != list(i2):
        return False
    return all(torch.equal(x, y) for x, y in zip(b1[:3] + [ws_view(b1[3], i1)], b2[:3] + [ws_view(b2[3], i2)]))
