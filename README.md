# GPU Accelerated Nested Sampling

## Abstract

Nested Sampling is a Monte Carlo method that performs parameter estimation and model comparison robustly for a variety of high dimensional and complicated distributions. It has seen widespread usage in the physical sciences, however in recent years increasingly it is viewed as part of a legacy code base, with GPU native paradigms such as neural simulation based inference coming to the fore. In this work we demonstrate that we can effectively reformulate Nested Sampling to a form that is highly amenable to modern GPU hardware, taking unique advantage of vectorization opportunities to accelerate numerical inference to state of the art levels. We provide a public implementation of this code, distributed via the blackjax statistical framework, which allows direct comparison with other well-established statistical methods such as Hamiltonian Monte Carlo and Sequential Monte Carlo, and in this contribution will explore its application to a number of inference problems such as Gravitational Wave parameter estimation and CMB cosmology.

 
[[PDF](https://github.com/williamjameshandley/talks/raw/ini_2025/will_handley_ini_2025.pdf)] 
