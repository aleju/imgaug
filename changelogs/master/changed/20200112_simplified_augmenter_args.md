# Simplified Standard Parameters of Augmenters #567 #595

Changed the standard parameters shared by all augmenters to a
reduced and more self-explanatory set. Previously, all augmenters
shared the parameters `name`, `random_state` and `deterministic`.
The new parameters are `seed` and `name`.

`deterministic` was removed as it was hardly ever used and because
it caused frequently confusion with regards to its meaning. The
parameter is still accepted but will now produce a deprecation
warning. Use `<augmenter>.to_deterministic()` instead.
(Reminder: `to_deterministic()` is necessary if you want to get
the same samples in consecutive augmentation calls. It is *not*
necessary if you want your generated samples to be dependent on
an initial seed or random state as that is *always* the case
anyways. You only have to manually set the seed, either
augmenter-specific via the `seed` parameter or global via
`imgaug.random.seed()` (affects only augmenters without their
own seed).)

`random_state` was renamed to `seed` as providing a seed value
is the more common use case compared to providing a random state.
Many users also seemed to be unaware that `random_state` accepted
seed values. The new name should make this more clear.
The old parameter `random_state` is still accepted, but will
likely be deprecated in the future.

**[breaking]** This patch breaks if one relied on the order of
`name`, `random_state` and `deterministic`. The new order is now
`seed=..., name=..., random_state=..., deterministic=...` (with the
latter two parameters being outdated or deprecated)
as opposed to previously
`name=..., deterministic=..., random_state=...`.
