### Log Book

Documented below are my general findings from implementing the algorithms.

## 04/26 - `vanilla_policy_grad.py`

- Implemented barebones, full return (not reward-to-go) policy gradient
- Overall each e2e run is still pretty random, variance is high
- One helpful trick was to normalize the gradients with `batch_steps` rather than `num_trajectories`
  - Although the latter is the correct expectation sample 
- Serendipitously MLX uses functional approach to autograd
    - Makes it easier to grab and apply the gradients
    - Need to wrap loss though: `grad_fn = mx.value_and_grad(loss_fn)`
- Initial runs failed due to a subtle bug in the implementation:
    - `(p + pt) * discounted_return` --> `p + (pt * discounted_return)`
    - Luckily monitoring the gradient norms showed me the explosions when increasing steps
- Increasing the number of trajectories in a batch does not improve the convergence much