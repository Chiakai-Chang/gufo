"""SHQ-T16 quantization and packing.

Implements the normative SHQ4-T16 v1 contract and the SHQ8-T16 encoding from
docs/QUANTIZATION.md. Byte layouts are byte-exact by design:

SHQ4 U4Z G64:
  qweight[n_tile][k_group][k16_subtile][output_lane=16][packed_k=8]
  scales [n_tile][k_group][output_lane=16]  BF16
  zeros  [n_tile][k_group][output_lane=16]  packed UINT4

Weight byte offset:
  (((n_tile * k_group_count + k_group) * k16_per_group + k16_subtile)
       * 16 + output_lane) * 8 + k_pair

Scale/zero index:
  scale_index = (n_tile * k_group_count + k_group) * 16 + output_lane

Quantizer rounding:
  q_real = w / s + z      (U4Z)
  q = clamp(round_to_nearest_ties_to_even(q_real), code_min, code_max)
  scale rounded to BF16 RNE-ties-to-even before final code selection.

This is the deterministic v1 scale search: per-channel-per-group min/max range
for U4Z, max-abs for S4/SHQ8. Imatrix/GPTQ-style search is a later recipe step.
"""

from __future__ import annotations

import numpy as np

U4_MAX = 15
S4_MIN, S4_MAX = -8, 7
S8_MAX = 127


def f32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (round-to-nearest, ties-to-even). Returns uint16 array."""
    a = arr.astype(np.float32, copy=False)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = np.right_shift(u, 16) & np.uint64(1)
    rounding_bias = np.uint64(0x7FFF) + lsb
    u = u + rounding_bias
    out = np.right_shift(u, np.uint64(16)).astype(np.uint32).astype(np.uint16)
    return out


def bf16_uint16_to_f32(ui: np.ndarray) -> np.ndarray:
    """Decode BF16 uint16 -> float32."""
    u32 = ui.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _round_even(v):
    # numpy round is banker's rounding (ties-to-even) already.
    return np.round(v)


def quantize_shq4(W: np.ndarray, group_size: int, symmetric: bool = False):
    """Quantize logical W[N,K] (float32) to SHQ4-T16 planes.

    Returns dict with byte planes: 'weight', 'scale', 'zero' and metadata.
    """
    W = W.astype(np.float32, copy=False)
    N, K = W.shape
    Np = (N + 15) // 16 * 16
    Kp = (K + group_size - 1) // group_size * group_size
    n_tiles = Np // 16
    k_groups = Kp // group_size
    k16_per = group_size // 16
    k_group_count = k_groups

    weight = bytearray(n_tiles * k_group_count * k16_per * 16 * 8)
    scale = bytearray(n_tiles * k_group_count * 16 * 2)  # BF16
    zero = bytearray(n_tiles * k_group_count * 16 // 2) if not symmetric else bytearray()

    # Pad W to Np x Kp with q=z (U4Z) / 0 (S4)
    Wp = np.zeros((Np, Kp), dtype=np.float32)
    Wp[:N, :K] = W

    for nt in range(n_tiles):
        for g in range(k_groups):
            col = g * group_size
            for lane in range(16):
                row = nt * 16 + lane
                w = Wp[row, col:col + group_size].astype(np.float32)
                wmin = float(w.min())
                wmax = float(w.max())
                s32 = 0.0
                z32 = 0.0
                q = np.zeros(group_size, dtype=np.float32)
                if wmax != wmin:
                    if symmetric:
                        s32 = max(abs(wmax), abs(wmin)) / 7.0
                    else:
                        s32 = (wmax - wmin) / 15.0
                # Round scale to BF16 RNE-ties-to-even before code selection.
                s_bf16 = float(bf16_uint16_to_f32(f32_to_bf16_uint16(np.array([s32])))[0])
                if s_bf16 > 0:
                    if symmetric:
                        q = _clamp(_round_even(w / s_bf16), S4_MIN, S4_MAX)
                        q = np.where(w == 0, 0, q)
                    else:
                        z32 = _clamp(_round_even(-wmin / s_bf16), 0, U4_MAX)
                        q = _clamp(_round_even(w / s_bf16 + z32), 0, U4_MAX)
                # else all-zero group: s=0, z=0, q=0

                # Pack scales + zeros (per lane)
                s_u16 = int(f32_to_bf16_uint16(np.array([s32]))[0])
                gidx = nt * k_group_count + g
                scale[gidx * 32 + lane * 2: gidx * 32 + lane * 2 + 2] = s_u16.to_bytes(2, "little")
                if not symmetric:
                    z_byte = int(z32)
                    pair = lane // 2
                    nib = (lane % 2 == 0)  # even lane in low nibble
                    zi = gidx * 8 + pair
                    if nib:
                        zero[zi] |= (z_byte & 0xF)
                    else:
                        zero[zi] |= (z_byte & 0xF) << 4
                # Pack weight microtile
                for k16 in range(k16_per):
                    base = col + k16 * 16
                    sub = w[base - col: base - col + 16].astype(np.float32)
                    qsub = q[base - col: base - col + 16]
                    micro = (gidx * k16_per + k16) * 16 * 8 + lane * 8
                    for kp in range(8):
                        v0 = int(qsub[2 * kp])
                        v1 = int(qsub[2 * kp + 1])
                        if symmetric:
                            v0 = (v0 & 0xF) if v0 >= 0 else ((v0 + 16) & 0xF)
                            v1 = (v1 & 0xF) if v1 >= 0 else ((v1 + 16) & 0xF)
                            b = (v0 & 0xF) | ((v1 & 0xF) << 4)
                        else:
                            b = (v0 & 0xF) | ((v1 & 0xF) << 4)
                        weight[micro + kp] = b

    return {
        "weight": bytes(weight),
        "scale": bytes(scale),
        "zero": bytes(zero),
        "N": N, "K": K, "Np": Np, "Kp": Kp,
        "group_size": group_size,
        "symmetric": symmetric,
        "format": f"SHQ4_T16_V1_{'S4' if symmetric else 'U4Z'}_G{group_size}",
    }


def quantize_shq8(W: np.ndarray, group_size: int):
    """Quantize W[N,K] to SHQ8-T16 (signed INT8, no zero point)."""
    W = W.astype(np.float32, copy=False)
    N, K = W.shape
    Np = (N + 15) // 16 * 16
    Kp = (K + group_size - 1) // group_size * group_size
    n_tiles = Np // 16
    k_groups = Kp // group_size
    k16_per = group_size // 16

    weight = bytearray(n_tiles * k_groups * k16_per * 16 * 16)  # 256 bytes/microtile
    scale = bytearray(n_tiles * k_groups * 16 * 2)
    Wp = np.zeros((Np, Kp), dtype=np.float32)
    Wp[:N, :K] = W
    for nt in range(n_tiles):
        for g in range(k_groups):
            col = g * group_size
            for lane in range(16):
                w = Wp[nt * 16 + lane, col:col + group_size].astype(np.float32)
                amax = float(np.max(np.abs(w)))
                s32 = amax / S8_MAX if amax > 0 else 0.0
                s_bf16 = float(bf16_uint16_to_f32(f32_to_bf16_uint16(np.array([s32])))[0])
                q = np.zeros(group_size, dtype=np.float32)
                if s_bf16 > 0:
                    q = _clamp(_round_even(w / s_bf16), -S8_MAX, S8_MAX)
                s_u16 = int(f32_to_bf16_uint16(np.array([s32]))[0])
                gidx = nt * k_groups + g
                scale[gidx * 32 + lane * 2: gidx * 32 + lane * 2 + 2] = s_u16.to_bytes(2, "little")
                for k16 in range(k16_per):
                    qsub = q[k16 * 16: k16 * 16 + 16]
                    micro = (gidx * k16_per + k16) * 16 * 16 + lane * 16
                    for kp in range(16):
                        weight[micro + kp] = int(qsub[kp]) & 0xFF
    return {
        "weight": bytes(weight),
        "scale": bytes(scale),
        "zero": b"",
        "N": N, "K": K, "Np": Np, "Kp": Kp,
        "group_size": group_size,
        "symmetric": False,
        "format": f"SHQ8_T16_V1_G{group_size}",
    }


def dequant_shq4(planes: dict) -> np.ndarray:
    """Decode SHQ4-T16 planes back to float32 W[Np,Kp]. CPU reference oracle."""
    weight = np.frombuffer(planes["weight"], dtype=np.uint8)
    scale = np.frombuffer(planes["scale"], dtype=np.uint16)
    Np, Kp = planes["Np"], planes["Kp"]
    G = planes["group_size"]
    n_tiles = Np // 16
    k_groups = Kp // G
    k16_per = G // 16
    k_group_count = k_groups
    W = np.zeros((Np, Kp), dtype=np.float32)
    zero_arr = np.frombuffer(planes["zero"], dtype=np.uint8) if planes["zero"] else None
    for nt in range(n_tiles):
        for g in range(k_groups):
            gidx = nt * k_group_count + g
            for lane in range(16):
                s = float(bf16_uint16_to_f32(np.array([scale[gidx * 16 + lane]]))[0])
                z = 0
                if zero_arr is not None:
                    zi = gidx * 8 + lane // 2
                    z = int(zero_arr[zi] & 0xF) if lane % 2 == 0 else int((zero_arr[zi] >> 4) & 0xF)
                for k16 in range(k16_per):
                    micro = (gidx * k16_per + k16) * 16 * 8 + lane * 8
                    for kp in range(8):
                        b = int(weight[micro + kp])
                        v0 = b & 0xF
                        v1 = (b >> 4) & 0xF
                        if planes["symmetric"]:
                            v0 = v0 if v0 < 8 else v0 - 16
                            v1 = v1 if v1 < 8 else v1 - 16
                        else:
                            v0 = v0 - z
                            v1 = v1 - z
                        row = nt * 16 + lane
                        col = g * G + k16 * 16 + 2 * kp
                        W[row, col] = s * v0
                        W[row, col + 1] = s * v1
    return W


def dequant_shq8(planes: dict) -> np.ndarray:
    weight = np.frombuffer(planes["weight"], dtype=np.uint8)
    scale = np.frombuffer(planes["scale"], dtype=np.uint16)
    Np, Kp = planes["Np"], planes["Kp"]
    G = planes["group_size"]
    n_tiles = Np // 16
    k_groups = Kp // G
    k16_per = G // 16
    W = np.zeros((Np, Kp), dtype=np.float32)
    for nt in range(n_tiles):
        for g in range(k_groups):
            gidx = nt * k_groups + g
            for lane in range(16):
                s = float(bf16_uint16_to_f32(np.array([scale[gidx * 16 + lane]]))[0])
                for k16 in range(k16_per):
                    micro = (gidx * k16_per + k16) * 16 * 16 + lane * 16
                    for kp in range(16):
                        v = int(weight[micro + kp])
                        if v >= 128:
                            v -= 256
                        W[nt * 16 + lane, g * G + k16 * 16 + kp] = s * v
    return W
