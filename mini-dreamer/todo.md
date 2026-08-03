I am seeing big slow downs during data loading
- The memmap is extremely expensive for large batch sizes
  - the encode function call is *much* slower than the diffusion **train step** - how??
  - It seems like things are not being fused
  - even when fused encoding takes ~**7x** more than jit_train_step!
#### Lets just cache the latents in the beginning and save either in RAM or storage
