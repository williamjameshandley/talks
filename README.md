# PolySwyft: a sequential simulation-based nested sampler 
[Kilian Scheutwinkel, Will Handley & Christoph Weniger]

We present recent advances in combining Nested Sampling and Simulation Based Inference. Nested sampling [1,2] is a numerical Bayesian inference tool for performing model comparison and parameter estimation, and has been widely applied across cosmology & particle physics. At its heart, the nested sampling meta-algorithm requires sampling from the prior subject to a hard likelihood constraint. It is therefore challenging to apply nested sampling if the likelihood is not explicitly available, as is the case in simulation based inference.

To address this, we propose a sequential algorithm which iteratively runs nested sampling on a ratio estimator whose training data are progressively informed by the dead points of the previous nested sampling runs. This replaces truncation with nested sampling's dead measure, and allows training of neural ratio estimators without marginalisation for generative models with around 10 free parameters without marginalisation.

As an example of this synthesis, we use PolyChord [3] for nested sampling and swyft [4] for the neural ratio estimator (forming PolySwyft), but emphasise the generality of this approach throughout. We showcase it on CMB and 21cm toy examples, and discuss potential future extensions to higher numbers of parameters and a deeper integration of the two algorithms.

- [1] https://arxiv.org/abs/2205.15570
- [2] https://arxiv.org/abs/2101.09675
- [3] https://github.com/PolyChord/PolyChordLite
- [4] https://swyft.readthedocs.io/en/stable/
[Paper to be released at around the time of the talk, material will have been presented at EuCAIFCon]

[[PDF](https://github.com/williamjameshandley/talks/raw/phystat_2024/will_handley_phystat_2024.pdf)] 
