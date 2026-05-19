rm -rf /tmp/jax_sdpa_probe

XLA_FLAGS="--xla_dump_to=/tmp/jax_sdpa_probe --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=optimized" \
uv run python - <<'PY'
import jax
import jax.numpy as jnp

print("jax:", jax.__version__)
print("devices:", jax.devices())

B = 8
S = 916   # ARC TRM seq_len 900 + puzzle prefix 16
H = 8
D = 64    # d_model 512 / 8 heads

@jax.jit
def run(q, k, v):
    return jax.nn.dot_product_attention(
        q, k, v,
        implementation="cudnn",
    )

q = jnp.ones((B, S, H, D), dtype=jnp.bfloat16)
k = jnp.ones((B, S, H, D), dtype=jnp.bfloat16)
v = jnp.ones((B, S, H, D), dtype=jnp.bfloat16)

y = run(q, k, v)
y.block_until_ready()
print("output:", y.shape, y.dtype)
PY

grep -RniE "custom-call|cudnn|fmha|flash|scaled_dot|dot_product_attention|softmax|stablehlo.dot| dot\\(" /tmp/jax_sdpa_probe | head -120