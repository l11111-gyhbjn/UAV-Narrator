import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEnc(nn.Module):
    def __init__(self, scales=(1, 2, 4), mode="temporal", spatial_tokens=169):
        super().__init__()
        assert mode in ("temporal", "flat")
        self.scales = list(scales)
        self.mode = mode
        self.spatial_tokens = spatial_tokens
        self.alpha = nn.Parameter(torch.zeros(len(self.scales)))
        self.scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _pool_resample(seq, s):
        L = seq.shape[-1]
        if s == 1:
            return seq
        k = min(s, L)
        pooled = F.avg_pool1d(seq, kernel_size=k, stride=k)
        return F.interpolate(pooled, size=L, mode="linear", align_corners=False)

    def forward(self, x):
        B, N, D = x.shape
        w = F.softmax(self.alpha, dim=0)
        if self.mode == "flat":
            seq = x.permute(0, 2, 1)
            out = sum(w[i] * self._pool_resample(seq, s) for i, s in enumerate(self.scales))
            out = out.permute(0, 2, 1)
        else:
            hw = self.spatial_tokens
            if N % hw != 0:
                raise ValueError(f"MS-Enc(temporal): {N} visual tokens is not a multiple of "
                                 f"{hw} spatial tokens; set --spatial-tokens for this backbone")
            t = N // hw
            seq = x.view(B, t, hw, D).permute(0, 2, 3, 1).reshape(B * hw, D, t)
            out = sum(w[i] * self._pool_resample(seq, s) for i, s in enumerate(self.scales))
            out = out.view(B, hw, D, t).permute(0, 3, 1, 2).reshape(B, N, D)
        return x + torch.sigmoid(self.scale) * out

class AttrCond(nn.Module):

    def __init__(self, num_attrs=23, hidden_dim=4096, num_prefix_tokens=4):
        super().__init__()
        self.num_prefix = num_prefix_tokens
        self.embed = nn.Sequential(
            nn.Linear(num_attrs, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim * num_prefix_tokens),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for m in self.embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, visual_tokens, attr_vec):
        B, N, D = visual_tokens.shape
        prefix = self.norm(self.embed(attr_vec.float()).view(B, self.num_prefix, D))
        return torch.cat([prefix.to(visual_tokens.dtype), visual_tokens], dim=1)

def init_module(m):
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.MultiheadAttention):
            if mod.in_proj_weight is not None:
                nn.init.xavier_uniform_(mod.in_proj_weight)
            if mod.in_proj_bias is not None:
                nn.init.zeros_(mod.in_proj_bias)

class TargetTokenCompressor(nn.Module):

    SELECT = ("box", "attn", "random", "uniform")

    def __init__(self, hidden_dim=4096, slices=5, grid=13, ring=1, max_fg=64, bg_grid=3,
                 zoom_grid=4, use_zoom=True, select="box", uniform_grid=5):
        super().__init__()
        assert select in self.SELECT, select
        assert select == "box" or not use_zoom, "the zoomed view needs the box (use select='box')"
        self.t, self.g, self.ring, self.max_fg = slices, grid, ring, max_fg
        self.bg_grid, self.zoom_grid, self.use_zoom = bg_grid, zoom_grid, use_zoom
        self.select, self.uniform_grid = select, uniform_grid
        D = hidden_dim
        z = lambda *s: nn.Parameter(torch.zeros(*s))
        self.e_fg, self.e_core, self.e_bg, self.e_zoom = z(D), z(D), z(D), z(D)
        self.pos_t, self.pos_h, self.pos_w = z(slices, D), z(grid, D), z(grid, D)
        self.pos_bg = z(bg_grid * bg_grid, D)
        self.pos_zoom = z(zoom_grid * zoom_grid, D)
        if select == "uniform":
            self.pos_uni = z(uniform_grid * uniform_grid, D)

    def select_mask(self, occ, cell_scores=None):
        keep = self.keep_mask(occ)
        if self.select == "box":
            return keep
        if self.select == "attn":
            assert cell_scores is not None, "select='attn' needs the [CLS] attention of the encoder"
            score = cell_scores.to(occ.device).float()
        else:
            gen = torch.Generator().manual_seed(int(occ.double().sum().item() * 1e4) % (2 ** 31))
            score = torch.rand(occ.shape, generator=gen).to(occ.device)
        out = torch.zeros_like(keep)
        for i in range(self.t):
            n = int(keep[i].sum())
            if n:
                out[i].view(-1)[torch.topk(score[i].flatten(), n).indices] = True
        return out

    def keep_mask(self, occ):
        core = (occ > 0).float()
        k = 2 * self.ring + 1
        ring = F.max_pool2d(core.unsqueeze(1), k, stride=1, padding=self.ring).squeeze(1)
        score = occ + 1e-3 * ring
        keep = torch.zeros_like(occ, dtype=torch.bool)
        for i in range(self.t):
            s = score[i].flatten()
            n = int((s > 0).sum())
            if n == 0:
                continue
            top = torch.topk(s, min(n, self.max_fg)).indices
            keep[i].view(-1)[top] = True
        return keep

    def forward(self, x, occ, zoom=None, cell_scores=None):
        B, N, D = x.shape
        assert B == 1, "TargetTokenCompressor works on one video at a time (batch size 1)"
        t, g = self.t, self.g
        assert N == t * g * g, f"expected {t * g * g} tokens, got {N}"
        x = x.view(t, g, g, D)
        if self.select == "uniform":
            k = self.uniform_grid
            out = []
            for i in range(t):
                u = F.adaptive_avg_pool2d(x[i].permute(2, 0, 1), k).permute(1, 2, 0).reshape(-1, D)
                out.append(u + self.e_bg + self.pos_t[i] + self.pos_uni)
            return torch.cat(out, 0).unsqueeze(0)
        occ = occ.to(x.device, x.dtype)
        keep = self.select_mask(occ, cell_scores)
        core_w = occ if self.select == "box" else torch.zeros_like(occ)
        dt = x.dtype
        out = []
        for i in range(t):
            m = (~keep[i]).to(dt)
            num = F.adaptive_avg_pool2d((x[i] * m[..., None]).permute(2, 0, 1), self.bg_grid)
            den = F.adaptive_avg_pool2d(m[None], self.bg_grid)
            bg = (num / den.clamp_min(1e-6)).permute(1, 2, 0).reshape(-1, D)
            valid = den.flatten() > 0
            bg = bg + self.e_bg + self.pos_t[i] + self.pos_bg
            out.append(bg[valid])
            r, c = torch.nonzero(keep[i], as_tuple=True)
            if len(r):
                fg = (x[i, r, c] + self.e_fg + core_w[i, r, c, None] * self.e_core
                      + self.pos_t[i] + self.pos_h[r] + self.pos_w[c])
                out.append(fg)
            if self.use_zoom and zoom is not None:
                zg = zoom.view(t, g, g, D)[i].permute(2, 0, 1)
                zt = F.adaptive_avg_pool2d(zg, self.zoom_grid).permute(1, 2, 0).reshape(-1, D)
                out.append(zt + self.e_zoom + self.pos_t[i] + self.pos_zoom)
        return torch.cat(out, 0).unsqueeze(0)

class TrajEncoder(nn.Module):

    def __init__(self, in_dim=7, d=256, steps=32, num_tokens=4, hidden_dim=4096, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(steps, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.zeros(num_tokens, d))
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        init_module(self)
        with torch.no_grad():
            nn.init.normal_(self.pos, std=0.02)
            nn.init.normal_(self.queries, std=0.02)

    def forward(self, traj):
        h = self.enc(self.inp(traj.float()) + self.pos)
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        p, _ = self.pool(q, h, h)
        return self.norm(self.out(p))
