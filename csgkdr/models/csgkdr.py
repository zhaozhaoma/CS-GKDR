from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GINEncoder, SFCNEncoder

MOD_ORDER = ("fmri", "smri", "clin")  # modality order used for the random choice of C


def _act(kind: str) -> nn.Module:
    return nn.ReLU(inplace=True) if kind == "relu" else nn.LeakyReLU(0.1, inplace=True)


def make_enc(in_d: int, out_d: int, drop: float, act: str = "relu") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, 256), nn.BatchNorm1d(256), _act(act), nn.Dropout(drop),
        nn.Linear(256, 128), _act(act), nn.Dropout(drop),
        nn.Linear(128, out_d),
    )


def enc_nodrop(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    for mod in seq:
        if isinstance(mod, nn.Dropout):
            continue
        if isinstance(mod, nn.BatchNorm1d):
            x = F.batch_norm(x, None, None, mod.weight, mod.bias, True, 0.0, mod.eps)
        else:
            x = mod(x)
    return x


def make_dec(c_dim: int, u_dim: int, out_d: int, drop: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(c_dim + u_dim, 256), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(256, out_d))


def make_head(in_d: int, drop: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(drop),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(drop * 0.5),
        nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(drop * 0.5),
        nn.Linear(64, 1),
    )


class CSGKDR(nn.Module):
    def __init__(self, m: dict, use_clin: bool = False, clin_indices: Optional[Sequence[int]] = None, clin_full_dim: int = 39,
                 reg_nodrop: bool = False):
        super().__init__()
        drop = m["dropout"]
        act = m.get("enc_activation", "relu")
        self.reg_nodrop = bool(reg_nodrop)
        dc, du = m["common_dim"], m["unique_dim"]
        smri_out = m["smri_channels"][-1]
        self.modalities: List[str] = ["fmri", "smri"] + (["clin"] if use_clin else [])
        # --- backbones ---
        self.fmri_backbone = GINEncoder(
            in_channels=m["fmri_in_channels"], hidden_channels=m["fmri_hidden"], num_layers=m["fmri_num_layers"],
            out_channels=m["fmri_out_dim"], dropout=m["fmri_dropout"], norm="batch", emb_style=m["fmri_emb_style"],
            num_nodes=m["fmri_num_nodes"] if m["fmri_emb_style"] != "none" else None, emb_size=m["fmri_emb_size"],
            use_attention=m["fmri_use_attention"], use_hierarchical_pooling=m["fmri_use_hierarchical_pooling"])
        self.smri_backbone = SFCNEncoder(channel_number=m["smri_channels"], use_attention=m["smri_use_attention"],
                                         use_depthwise_conv=m["smri_use_depthwise_conv"])
        h = {"fmri": m["fmri_out_dim"], "smri": smri_out}
        if use_clin:
            idx = list(range(clin_full_dim)) if clin_indices is None else list(clin_indices)
            self.register_buffer("clin_index", torch.tensor(idx, dtype=torch.long), persistent=False)
            self.clin_backbone = nn.Sequential(
                nn.Linear(len(idx), m["clin_hidden"]), nn.LayerNorm(m["clin_hidden"]), nn.ReLU(inplace=True), nn.Dropout(drop),
                nn.Linear(m["clin_hidden"], m["clin_out_dim"]), nn.LayerNorm(m["clin_out_dim"]), nn.ReLU(inplace=True), nn.Dropout(drop))
            h["clin"] = m["clin_out_dim"]
        for mod in self.modalities:
            setattr(self, f"{mod}_enc_c", make_enc(h[mod], dc, drop, act))
        for mod in self.modalities:
            setattr(self, f"{mod}_enc_u", make_enc(h[mod], du, drop, act))
        for mod in self.modalities:
            setattr(self, f"{mod}_dec", make_dec(dc, du, h[mod], drop))
        self.pred_head = make_head(len(self.modalities) * du + dc, drop)
        # predictor input order: trimodal [u_smri, u_clin, u_fmri, c]; bimodal [u_fmri, u_smri, c]
        self.head_order = ["smri", "clin", "fmri"] if use_clin else ["fmri", "smri"]

    def features(self, batch) -> Dict[str, torch.Tensor]:
        g = batch["fmri"]
        out = {"fmri": self.fmri_backbone(g).reshape(g.batch_size, -1)}
        smri = batch["smri"]
        _, _, D, H, W = smri.shape
        if D < 160 or H < 192 or W < 160:  # upsample small volumes to the SFCN input size
            smri = F.interpolate(smri, size=(max(D, 160), max(H, 192), max(W, 160)), mode="trilinear", align_corners=False)
        out["smri"] = self.smri_backbone(smri)
        if "clin" in self.modalities:
            out["clin"] = self.clin_backbone(batch["clin"].index_select(1, self.clin_index))
        return out

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        h = self.features(batch)
        out = {f"h_{m}": h[m] for m in self.modalities}
        cs = {m: getattr(self, f"{m}_enc_c")(h[m]) for m in self.modalities}
        M = len(self.modalities)
        if self.training:
            r = torch.rand(1).item()
            c = cs[self.modalities[min(int(r * M), M - 1)]]
        else:
            c = torch.stack([cs[m] for m in self.modalities], 0).mean(0)
        # unique encoders / decoders run in predictor-input order (this fixes the order of dropout draws)
        us = {m: getattr(self, f"{m}_enc_u")(h[m]) for m in self.head_order}
        for m in self.head_order:
            out[f"rec_{m}"] = getattr(self, f"{m}_dec")(torch.cat([cs[m], us[m]], 1))
        for m in self.modalities:
            out[f"c_{m}"] = cs[m]
            out[f"u_{m}"] = us[m]
        out["c"] = c
        out["pred"] = self.pred_head(torch.cat([us[m] for m in self.head_order] + [c], 1))
        if self.training and self.reg_nodrop:  # dropout-free codes for the anti-collapse terms
            for m in self.modalities:
                out[f"ureg_{m}"] = enc_nodrop(getattr(self, f"{m}_enc_u"), h[m])
                out[f"creg_{m}"] = enc_nodrop(getattr(self, f"{m}_enc_c"), h[m])
        return out
