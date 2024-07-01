# The scaling frontier of nested sampling 

## Abstract

There are two questions any afficionado of nested sampling~[1] should be able to answer:

*How long does nested sampling take?*

$$\boxed{T_\mathrm{NS} = T_\mathcal{L} \times n_\mathrm{live}  \times \mathcal{D}_\mathrm{KL}\times f_\mathrm{sampler}}, $$

*How accurate is nested sampling?*

$$ \boxed{\sigma_{\log\mathcal{Z}} \approx \sqrt{\mathcal{D}_\mathrm{KL}/n_\mathrm{live}}}$$

The first equation says that the expected runtime $T_\mathrm{NS}$ of nested sampling can be calculated as the product of the average likelihood call time $T_\mathcal{L}$, the number of live points $n_\mathrm{live}$, the Kullback Liebler divergence between the prior and posterior, and the efficiency of the sampler $f_\mathrm{sampler}$. 

The second equation says that the error on the evidence $\mathcal{Z}$ is approximately the square root of the KL divergence over the number of live points.

If a nested sampling user is able to explain why these relations take the form they do, then they are well on their way to mastery.
We shall go through each of these in turn, examining the runtime scaling and considerations which lie at the forefront of the field.

I will use this framework to discuss frontiers of nested sampling acting to relax the assumptions in the above, with the aim of reducing $T_\mathrm{NS}\sim\mathcal{O}(d^4)$ to a scaling appropriate to the ultra high dimensional regime such as neural networks, image processing and protein folding. Such frontiers include (but are not limited to) posterior repartitioning~[2], precision nested sampling~[3], reversible nested sampling and gradient-based nested sampling~[5].
\medskip\noindent{References: }

[1] John Skilling "Nested sampling for general Bayesian computation," Bayesian Analysis, Bayesian Anal. 1(4), 833-859, (December 2006)

[2] Chen, Xi, Farhan Feroz, and Michael Hobson. "Bayesian posterior repartitioning for nested sampling." Bayesian Analysis 18.3 (2023): 695-721.

[3] Petrosyan, Aleksandr, and Will Handley. "SuperNest: accelerated nested sampling applied to astrophysics and cosmology." Physical Sciences Forum. Vol. 5. No. 1. MDPI, 2023.

[4] Skilling, J. Galilean and Hamiltonian Monte Carlo. Proceedings 2019, 33, 19. 

[5] Lemos, Pablo, et al. "Improving gradient-guided nested sampling for posterior inference." arXiv preprint arXiv:2312.03911 (2023).

[[PDF](https://github.com/williamjameshandley/talks/raw/maxent_2024/will_handley_maxent_2024.pdf)] 
