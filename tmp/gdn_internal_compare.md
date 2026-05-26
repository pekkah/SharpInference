# GDN-internal compare, head 0, position 12 — sharpi vs llama.cpp

Run: 2026-05-26. Sharpi commit `8b7856b`, llama.cpp build-mini b9333 with
SHARPI-COMPARE extended (`ggml/src/ggml-cpu/ops.cpp` — gated_delta_net trace).
Both built from the same Q4_K_M GGUF (`Qwen3.6-27B-MTP-Q4_K_M.gguf`).

Format per layer:
```
[post-decay]   decay, b, S_l2, S_sum     — gate scalars + decayed previous-token state
[dvec]         l2, sum                    — GDN delta vector (b · (v − S^T·k))
[post-rank1]   S_l2, S_sum                — state after rank-1 update for this token
[p-readout]    l2, sum                    — q·S attention output (post 1/√d scale, pre-RMSNorm)
```

## L0 — BIT-EXACT

| field        | sharpi             | llama              | Δ           |
|--------------|--------------------|--------------------|-------------|
| decay        | 0.9389449          | 0.9389449          | 0           |
| b            | 0.1387825          | 0.1387825          | 0           |
| S_l2 (dec)   | 13.6203            | 13.6203            | 0           |
| S_sum (dec)  | 24.7786            | 24.7786            | 0           |
| dvec l2      | 3.4526             | 3.4526             | 0           |
| dvec sum     | 4.38459            | 4.38459            | 0           |
| S_l2 (rk1)   | 14.2949            | 14.2949            | 0           |
| S_sum (rk1)  | 26.7992            | 26.7991            | 0.0001 (4 ulp) |
| p-read l2    | 0.0560653          | 0.0560653          | 0           |
| p-read sum   | 0.0695616          | 0.0695616          | 0           |

L0 GDN op is **bit-exact** between sharpi and llama within FP rounding. Every internal
stage matches to printed precision.

## L20 — already drifted on the FIRST gate scalar

| field        | sharpi             | llama              | rel Δ       |
|--------------|--------------------|--------------------|-------------|
| decay        | 0.9405219          | 0.9397262          | 0.085 %     |
| b            | 0.706099           | 0.7066947          | 0.084 %     |
| S_l2 (dec)   | 0.971675           | 0.966291           | 0.56 %      |
| S_sum (dec)  | 0.089277           | 0.0864391          | 3.3 %       |
| dvec l2      | 0.171252           | 0.174656           | 1.95 %      |
| dvec sum     | 0.0156366          | 0.0128166          | 22 %  *     |
| S_l2 (rk1)   | 1.05771            | 1.05537            | 0.22 %      |
| S_sum (rk1)  | 0.0953709          | 0.0910986          | 4.7 %       |
| p-read l2    | 0.0055957          | 0.005444           | 2.8 %       |
| p-read sum   | 0.00160619         | 0.00152308         | 5.5 %       |

`*` Sums in the 0.01–0.1 range are sensitive to cancellation; relative is loud, absolute
is tiny (Δ=0.003).

## L40 — small input drift

| field        | sharpi             | llama              | rel Δ       |
|--------------|--------------------|--------------------|-------------|
| decay        | 0.9921878          | 0.9913852          | 0.08 %      |
| b            | 0.3909036          | 0.394094           | 0.81 %      |
| S_l2 (dec)   | 0.775718           | 0.774095           | 0.21 %      |
| S_sum (dec)  | 1.41563            | 1.41411            | 0.11 %      |
| dvec l2      | 0.0744988          | 0.0733382          | 1.58 %      |
| dvec sum     | 0.0740566          | 0.0767525          | 3.5 %       |

## L60 — substantial input drift

| field        | sharpi             | llama              | rel Δ       |
|--------------|--------------------|--------------------|-------------|
| decay        | 0.7430784          | 0.7601695          | 2.25 %      |
| b            | 0.8698069          | 0.8825179          | 1.44 %      |
| S_l2 (dec)   | 0.745456           | 0.756848           | 1.5 %       |
| S_sum (dec)  | 7.17178            | 7.42605            | 3.4 %       |
| dvec l2      | 0.752582           | 0.757727           | 0.68 %      |
| dvec sum     | -0.520291          | -0.598467          | 13 %        |
| S_l2 (rk1)   | 1.19592            | 1.20736            | 0.95 %      |
| S_sum (rk1)  | 8.50435            | 9.02975            | 5.8 %       |
| p-read l2    | 0.0222089          | 0.0218814          | 1.5 %       |
| p-read sum   | -0.0362503         | -0.0377764         | 4.0 %       |

## Conclusion

The **GDN op itself is bit-exact** at L0 (the inputs are identical → outputs are identical
to printed precision). At L20 the FIRST gate scalar to compute — `decay = exp(softplus(α+dt)·a)`
— already disagrees by 0.085%, which means **α (= ssm_alpha @ attn_norm)** at L20·pos12
already disagrees. α at L20 depends only on attn_norm(residual_L20) and the weight matrix,
so the residual stream at L20 has drifted upstream of the GDN op.

This **falsifies the GDN dot-order-at-deep-layers hypothesis** as the originator. Given the
same inputs, sharpi's GDN op produces the same bits as llama's. The drift seeps in
elsewhere — the residual-stream accumulation, attention QK^T softmax, MRoPE rotation, or
RmsNorm — before the alpha/beta matmuls of deeper GDN layers see their inputs.

## Next probe

To localize: dump attn_norm output and residual stream at positions ≤ 12 for each
layer (or just for layers ∈ {1, 3, 5, 7, 10, 15, 19}) and compare sharpi vs llama —
find the first layer where Δ(attn_norm-out) is non-trivially > L0's. That will pin
either (a) an attention layer's MRoPE / softmax / Q6_K matmul or (b) RmsNorm precision
as the originator. L3 attn-resid was previously confirmed bit-exact at sum-level
(Δsum 0.003 in 14.8) so the originator likely lies in L4 .. L19.
