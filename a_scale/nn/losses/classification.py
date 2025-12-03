import jax
import equinox as eqx
import jax.numpy as jnp

@eqx.filter_jit
def condcrossent(model, data):
    x, y = data
    logits = model(x)
    res = jnp.take_along_axis(-jax.nn.log_softmax(logits), y, None).squeeze()
    return res

@eqx.filter_jit
def condacc(model, data):
    x, y = data
    logits = model(x)
    acc = jnp.argmax(logits) == y
    return acc