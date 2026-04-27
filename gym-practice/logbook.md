### Log Book

Documented below are my general findings from implementing the algorithms.

## 04/26 - `reinforce.py`

- Implemented reward-to-go as simple cumsum in reverse
- Noticed that `max_episode_steps=100` gave results same as vanilla but bumping to 200 no longer causes instabilities
- Normalizing the gradient norm helps
- Interestingly when the initial state is good, we cant seem to build on it
    - The model is always worse after training
    - e.g. seed=4321 the initial reward is 60+ but ends near the 10s
- Fixed a very silly bug in reward accumulation

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
- Choosing 0.9 as a decent discount factor