# Sampling

## Sampling Methods Implemented
### 1. Inverse Transform
Define an Exponential prior for β. Then sample a number from a uniform distribution. Use inverse CDF to draw samples.

### 2. Accept Reject
Define a Beta(2,5)-shaped prior over (0,1), use a Uniform(0,1) proposal,
and perform rejection sampling. γ is mapped from (0,1) to its actual range.

### 3. Cluster
Create synthetic clusters, each with its own parameters and
population size. Cluster sampling can be used to estimate population-level epidemic
quantities.

### 4. MCMC
Given noisy I(t) observations, infer
β and γ using a random walk algorithm. The walker starts somewhere in log-log space. At each step, we talk a random walk. If we end up in a more likely location, it is accepted immediately. If we end up in a less likely location, we may still accept it to avoid getting stuck. Over time, we spend a lot of time in high likelihood regions and less time in low likelihood regions. 

