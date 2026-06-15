#!/usr/bin/env python3
"""
Reference oracle for the Gemma 4 "unified" (encoder-free) vision projector
(clip.vision.projector_type = "gemma4uv").

Faithfully reimplements tools/mtmd/models/gemma4uv.cpp from llama.cpp using the
real mmproj tensors, so the C# GemmaUvVisionEmbedder can be parity-checked against
it (issue #250).

Forward (per gemma4uv.cpp::build):
    inp_raw [W,H,3] (planar CHW, values 0..1)
    im2col(patch=48, stride=48) -> per patch a 6912 vector laid out [c*48*48 + ky*48 + kx]
    LayerNorm(patch_norm_1, eps=1e-5)           # 6912-dim, BEFORE the linear (affine-invariant
                                                #  -> the missing *2-1 rescale is irrelevant)
    matmul(v.patch_embd.weight: 6912->3840) + v.patch_embd.bias
    LayerNorm(patch_norm_2, eps=1e-5)           # 3840-dim
    + pos_x[col] + pos_y[row]   (col=i%n_cols, row=i//n_cols, n_cols=W/48)
    LayerNorm(patch_norm_3, eps=1e-5)           # "pos_norm"
    rms_norm(eps=1e-6)                           # embedding_pre_projection_norm, NO weight
    matmul(mm.input_projection.weight: 3840->3840)   # bf16
  -> [n_tokens=(W/48)*(H/48), 3840]

Usage:
    PYTHONPATH=C:/p/llama.cpp/gguf-py python scripts/gemma4uv_ref.py \
        E:/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf [out_dir]

Writes (default out_dir = tests/fixtures/gemma4uv):
    input_chw.f32    raw float32, shape [3,H,W] (the synthetic preprocessed image)
    output.f32       raw float32, shape [n_tokens,3840] (projector soft tokens)
    meta.json        shapes, dims, per-step stats
"""
import sys, os, json, struct
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 48          # effective patch (config patch_size 16 * n_merge 3) for gemma4uv
LN_EPS = 1e-5       # gemma4uv.cpp hardcodes pytorch-LayerNorm default eps
RMS_EPS = 1e-6      # hparams.eps == clip.vision.attention.layer_norm_epsilon

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "E:/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/gemma4uv"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(mmproj)
    tmap = {t.name: t for t in r.tensors}

    def raw_f32(name):
        """Return the tensor as a flat float32 array (dequantizing bf16)."""
        t = tmap[name]
        tt = t.tensor_type.name
        if tt == "F32":
            return np.asarray(t.data, dtype=np.float32).reshape(-1)
        if tt == "BF16":
            bits = np.frombuffer(t.data.tobytes(), dtype=np.uint16).astype(np.uint32)
            return (bits << 16).view(np.float32)
        raise RuntimeError(f"unhandled dtype {tt} for {name}")

    def ne(name):
        # gguf stores ne fastest-first; ReaderTensor.shape echoes that order.
        return [int(x) for x in tmap[name].shape]

    def mat(name):
        """Linear weight with ne=[in,out] -> numpy (out,in) so y = x @ W.T."""
        n = ne(name); assert len(n) == 2, n
        nin, nout = n[0], n[1]
        return raw_f32(name).reshape(nout, nin)

    def vec(name):
        return raw_f32(name)

    # ---- tensors ----
    pe_w = mat("v.patch_embd.weight")          # (3840, 6912)
    pe_b = vec("v.patch_embd.bias")            # (3840,)
    n1w, n1b = vec("v.patch_norm.1.weight"), vec("v.patch_norm.1.bias")   # 6912
    n2w, n2b = vec("v.patch_norm.2.weight"), vec("v.patch_norm.2.bias")   # 3840
    n3w, n3b = vec("v.patch_norm.3.weight"), vec("v.patch_norm.3.bias")   # 3840
    mm_w = mat("mm.input_projection.weight")   # (3840, 3840)
    # position table: ne=[3840,1120,2] -> reshape (2,1120,3840); [0]=x table, [1]=y table
    pne = ne("v.position_embd.weight")
    n_embd, pos_size = pne[0], pne[1]
    pos = raw_f32("v.position_embd.weight").reshape(pne[2], pos_size, n_embd)
    tbl_x, tbl_y = pos[0], pos[1]              # (1120, 3840) each

    assert pe_w.shape == (3840, 6912), pe_w.shape
    assert mm_w.shape == (3840, 3840), mm_w.shape

    def layernorm(x, w, b, eps):               # ggml_norm over last axis + affine
        m = x.mean(-1, keepdims=True)
        d = x - m
        v = (d * d).mean(-1, keepdims=True)
        return d / np.sqrt(v + eps) * w + b

    def rmsnorm(x, eps):                        # ggml_rms_norm, no weight
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

    # ---- synthetic preprocessed image: CHW [3,H,W], values 0..1, deterministic ----
    H, W = 288, 384                            # -> (8 x 6) = 48 tokens (within [40,280])
    gy, gx = H // PATCH, W // PATCH
    n_tok = gy * gx
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()   # CHW [3,H,W] in [0,1]

    # ---- im2col: P[p, c*PATCH*PATCH + ky*PATCH + kx] ----
    P = np.empty((n_tok, 3 * PATCH * PATCH), dtype=np.float32)
    for p in range(n_tok):
        pr, pc = p // gx, p % gx
        block = img[:, pr*PATCH:(pr+1)*PATCH, pc*PATCH:(pc+1)*PATCH]   # (3,48,48), order c,ky,kx
        P[p] = block.reshape(-1)

    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("im2col", P)

    x = layernorm(P, n1w, n1b, LN_EPS);                 rec("after_norm1", x)
    x = x @ pe_w.T + pe_b;                               rec("after_patch_embd", x)
    x = layernorm(x, n2w, n2b, LN_EPS);                 rec("after_norm2", x)
    n_cols = W // PATCH
    px = np.array([i % n_cols for i in range(n_tok)])
    py = np.array([i // n_cols for i in range(n_tok)])
    x = x + tbl_x[px] + tbl_y[py];                       rec("after_pos", x)
    x = layernorm(x, n3w, n3b, LN_EPS);                 rec("after_norm3", x)
    x = rmsnorm(x, RMS_EPS);                             rec("after_rms", x)
    out = x @ mm_w.T;                                    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta = dict(patch=PATCH, H=H, W=W, gx=gx, gy=gy, n_tokens=n_tok, n_embd=n_embd,
                ln_eps=LN_EPS, rms_eps=RMS_EPS, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_tok},{n_embd}] to {out_dir}")

if __name__ == "__main__":
    main()
