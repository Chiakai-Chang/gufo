"""Shape-specialized MiniMax H3 dense attention for AMD gfx1151."""

import triton
import triton.language as tl


@triton.jit
def _h3_attention_inner(
    accumulator,
    running_sum,
    running_max,
    query_values,
    key,
    value,
    sequence,
    head_base,
    start_block,
    block_count,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MASK_KEY_TAIL: tl.constexpr,
):
    key_rows = tl.arange(0, BLOCK_N)
    dimensions = tl.arange(0, HEAD_DIM)
    for block_index in range(block_count):
        current_key_rows = (start_block + block_index) * BLOCK_N + key_rows
        key_offsets = (
            head_base
            + dimensions[:, None]
            + current_key_rows[None, :] * HEAD_DIM
        )
        if MASK_KEY_TAIL:
            key_values = tl.load(
                key + key_offsets,
                mask=current_key_rows[None, :] < sequence,
                other=0.0,
            )
        else:
            key_values = tl.load(key + key_offsets)
        value_offsets = (
            head_base
            + current_key_rows[:, None] * HEAD_DIM
            + dimensions[None, :]
        )
        if MASK_KEY_TAIL:
            value_values = tl.load(
                value + value_offsets,
                mask=current_key_rows[:, None] < sequence,
                other=0.0,
            )
        else:
            value_values = tl.load(value + value_offsets)

        scores = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        if MASK_KEY_TAIL:
            scores = tl.where(
                current_key_rows[None, :] < sequence,
                scores,
                -float("inf"),
            )
        scores += QK_SCALE * tl.dot(query_values, key_values)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        scores -= next_max[:, None]
        probabilities = tl.math.exp2(scores)
        block_sum = tl.sum(probabilities, axis=1)
        correction = tl.math.exp2(running_max - next_max)
        accumulator *= correction[:, None]
        running_sum = running_sum * correction + block_sum
        running_max = next_max
        accumulator = tl.dot(
            probabilities.to(value_values.dtype),
            value_values,
            accumulator,
        )
    return accumulator, running_sum, running_max


@triton.jit
def h3_attention_forward(
    query,
    key,
    value,
    output,
    sequence,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    head_base = head * sequence * HEAD_DIM
    query_offsets = (
        head_base
        + query_rows[:, None] * HEAD_DIM
        + dimensions[None, :]
    )
    query_values = tl.load(
        query + query_offsets,
        mask=query_rows[:, None] < sequence,
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], -3.40282e38, tl.float32)
    running_sum = tl.full([BLOCK_M], 1.0, tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    full_blocks = sequence // BLOCK_N
    accumulator, running_sum, running_max = _h3_attention_inner(
        accumulator,
        running_sum,
        running_max,
        query_values,
        key,
        value,
        sequence,
        head_base,
        0,
        full_blocks,
        QK_SCALE,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        MASK_KEY_TAIL=False,
    )
    if full_blocks * BLOCK_N < sequence:
        accumulator, running_sum, running_max = _h3_attention_inner(
            accumulator,
            running_sum,
            running_max,
            query_values,
            key,
            value,
            sequence,
            head_base,
            full_blocks,
            1,
            QK_SCALE,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            MASK_KEY_TAIL=True,
        )

    accumulator *= 1.0 / running_sum[:, None]
    output_offsets = (
        query_rows[:, None] * OUTPUT_HEAD_STRIDE
        + head * HEAD_DIM
        + dimensions[None, :]
    )
    tl.store(
        output + output_offsets,
        accumulator.to(tl.bfloat16),
        mask=query_rows[:, None] < sequence,
    )
