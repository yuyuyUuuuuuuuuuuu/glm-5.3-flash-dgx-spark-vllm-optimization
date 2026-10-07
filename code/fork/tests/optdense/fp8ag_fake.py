"""opt-dense test fixture: a decoder layer whose forward mirrors the SP-patched production lines around the attention
gather (the frame/line detection of fp8_w8a8._sp_all_gather_fp8 reads THIS file's source) and KDA-like attention
classes (one that passes the AST check, one that does not)."""
import torch

MHC_SP_ACTIVE = True
sp_all_gather = None          # the test installs the wrapper here


class GoodAttn(torch.nn.Module):
    def __init__(self, in_proj):
        super().__init__()
        self.in_proj_qkvbfg_a = in_proj

    def forward(self, hidden_states, positions):
        num_tokens = hidden_states.size(0)
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        out = torch.empty((num_tokens, 8), dtype=hidden_states.dtype, device=hidden_states.device)
        out.zero_()
        return projected


class BadAttn(GoodAttn):
    def forward(self, hidden_states, positions):
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        return projected + hidden_states.sum()


class FakeDecoder(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn
        self.layer_idx = 0

    def forward(self, positions, x):
        if MHC_SP_ACTIVE:
            # a comment line between the gather condition and the gather, as in production
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )
        y = x
        x = x[:, :64].contiguous()
        if MHC_SP_ACTIVE:
            x = sp_all_gather(x)[: positions.shape[0]]

        if MHC_SP_ACTIVE:
            z = self.self_attn.in_proj_qkvbfg_a.weight
        return y, x
