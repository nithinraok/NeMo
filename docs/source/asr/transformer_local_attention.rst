Offline Transformer locality experiments
=======================================

``TransformerEncoder`` supports two optional locality controls. Defaults preserve
the original parameter names, shapes, and full-attention behavior.

Mixed global and local heads
----------------------------

With 16 heads, ``local_attention_heads: 4`` restricts the final four heads to
``local_attention_context: [4, 4]``. The other twelve heads remain global. All
heads retain their existing projections and per-head dimensions. Padding and any
base causal constraints apply to every head. The window includes the query:
left four + query + right four = nine encoder frames after subsampling.

An additional narrow attention stage
-----------------------------------

Set ``extra_attention_dim: 320`` and ``extra_attention_heads: 4`` for a second
attention stage in each block of a 1280-wide, 16-head encoder. Its Q/K/V
projections map 1280 to 320 each; its output projection maps 320 to 1280. The
block order is global attention, added pre-norm residual attention, existing
FFN. The original FFN is unchanged. No GLU or convolution is introduced.

``extra_attention_context: [4, 4]`` makes this added stage local. ``null`` makes
it global, providing an equal-capacity control. Added attention currently
supports RoPE and no positional encoding, and requires the base per-head width
(80 in this example), allowing RoPE buffers to be shared.

``extra_attention_seed`` defaults to 1729. Added modules initialize in a separate
CPU RNG scope after the shared encoder, preserving shared model initialization
and subsequent decoder initialization when the outer model seed is fixed.
The AED training entrypoint accepts a top-level ``seed`` (e.g., 42) to seed
Lightning and worker initialization before model construction.
Local and global added-stage controls initialize identically.

Execution and streaming
-----------------------

These options currently use padded offline execution. Sequence-packed execution
and ``StreamingTransformerEncoder`` reject them explicitly. This does not choose
a streaming chunk size. A streaming adapter must combine head-specific locality
with the stream's causal/chunk mask; an added attention stage also needs its own
intermediate-input cache. Do not reuse the original layer-input cache for it.
