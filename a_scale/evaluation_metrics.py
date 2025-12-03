import jax.numpy as jnp

def log_ASE(pred, actual):
    # filter for positions where both pred and actual are positive
    # and are not NaN
    pred_actual_filter = jnp.isfinite(pred) & jnp.isfinite(actual) & (pred > 0) & (actual > 0)
    pred = pred[pred_actual_filter]
    actual = actual[pred_actual_filter]
    #pred_actual = jnp.where((pred > 0) & (actual > 0), True, False)
   # pred = pred[pred_actual]
   # actual = actual[pred_actual]
    # average squared error on the log scale
    log_pred = jnp.log(pred)
    log_actual = jnp.log(actual)
    ASE = jnp.mean((log_pred - log_actual)**2)
    sqrt_ASE = jnp.sqrt(ASE)
    #return jnp.mean((log_pred - log_actual)**2) * 1000
    return sqrt_ASE * 100