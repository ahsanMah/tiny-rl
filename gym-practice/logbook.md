### Log Book

Documented below are my general findings from implementing the algorithms.

## 05/08 - Debugging Normalization
- State was being normalized but the policy was still gettng unnormalized observations during play
- The environment resetting was causing slight problems due to the next-step of gymnasium
  - fixed to same-step where the final observation is kept separate and the reset obs is returned on failure
- One big factor was the gradient explosion was occuring due to the normalization
  - Normalizing the  value function gradient norm via clipping seems to have fixed this!

## 05/07 - other Environments

- When switching to MountainCar or Acrobot, performance degrades
- My intial thoughts were that the scale of the state spaces was just too small
- Normalizing via a running mean / var is a possible solution
  - However this is leading to instabilities in CartPole testing


## 05/05 - Full Vectorization
- We can use gymnaisums vectorized environments
- Neural nets trivially take in batches but log_prob calculation needed to be modified
  - actions are batch of indices to select logits
  - mlx allows for easy selection via `mx.take_along_axis(log_prob, action[:, None], axis=1)`
- Sampling also needed to be modded to grab the categorical index
  - recall that basic procedure is:
    - sample unform val `[0,1)`
    - grab the 'bucket' according to probs (cumulative probabilities to be precise)
    - i.e return `catgeory if uniform_val >= CDF[category]`
  - Luckily I noticed that `mx.argmax` returns the first (left most) index
  - So you can just `argmax(random < cdf)`
- **Important**: my existing asserts for the Buffer helped!
  - Forced me to really think about my terminal values
  - When an episode/trajectory *truncates* your current action is still valid
  - thus you shoudl feed in value of next_state: `value(observation)`
  - Previously I was feeding `value(state)` which I believe was a subtle bug

## 05/02 - Thinking through vectorizing

- Cumuluative sum vecotrization can be tricky
- some options:
    - manual for loop
    - python accumulator (also for loop underneath but in C ...?)
    - cumsum(x * powers) / powers
```py
    rewards = [0 , 1, 1, 1]
    # rewards = self.reward[::-1]
    reward_to_go = []
    cumsum = 0
    # r[0] = gamma(0)r[0] + gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
    # r[1] =                gamma(0)r[1] + gamma(1)r[2] + gamma(2)r[3]
    for r in rewards:
        cumsum = r + gamma(1) * cumsum
        reward_to_go.append(cumsum)
    
    print("cumsum:", reward_to_go)
    reward_to_go = accumulate(
        rewards, lambda r_sum, rt: rt + gamma(1) * r_sum, initial=None
    )
    print("itertools:", list(reward_to_go))

    #cumsum / powers trick
    rewards = mx.array(rewards)
    gammas = mx.array([gamma(i) for i in range(len(rewards))])
    # (in reverse)
    # r[0] = gamma(0)r[0] + gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
    # r[1] =                gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
    rewards *= gammas
    reward_to_go = mx.cumsum(rewards) / gammas
    print("cumsum:", reward_to_go)
```


## 04/29

- Important note about terminal values
    - When the agent 'dies' after taking the step, the reward should be 0!
    - to make the logic work correctly, we should precompute advantages before policy updates

## 04/28 - `reinforce_gae.py`
- Implementing GAE variant for reinforce
 - Note that two value function calls re required for TD residual estimate
- Main trick is to compute advantages in reverse so that you can accumulate "future" advantages
- Value function is still trained on reward-to-go as before
- Yet another hyperparamter (ema)


## 04/26 - `reinforce.py`

- Implemented reward-to-go as simple cumsum in reverse
- Noticed that `max_episode_steps=100` gave results same as vanilla but bumping to 200 no longer causes instabilities
- Normalizing the gradient norm helps
- Interestingly when the initial state is good, we cant seem to build on it
    - The model is always worse after training
    - e.g. `seed=4321` the initial reward is 60+ but ends near the 10s
- Fixed a very silly bug in reward accumulation
- Value function can be trained with just using existing (state, reward) pairs
    - Required a small rewrite but is conceptually straightforward
    - Encountered silent bug when regressing against a global variable and getting 0 loss smh
- Baseline variant improves performance in `seed=4321`!
- Still difficult to know which hyperparameter has the most outstanding effect:
    - `max_episode_steps`, `lr`, `num_epochs`, `num_trajectories`, `discount_factor`

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
