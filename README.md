# Two Revolutions: GPUs for Science, AI for Scientists

## Abstract

Modern GPUs and large language models represent two distinct but concurrent shifts in how science is done. I will explore both, and suggest they may be more separable than current discourse implies.

GPU hardware—developed for machine learning—can accelerate classical statistical methods such as nested sampling by factors of 100–1000×, enabling rigorous Bayesian inference at previously impractical scales. I will present recent work from my group applying GPU-native nested sampling to CMB cosmology, cosmic shear, Type Ia supernovae, and gravitational wave parameter estimation.

Much of the narrative around AI in science focuses on foundation models or fully autonomous agents generating research end-to-end. I will present an alternative view: that large language models are most effective at reducing the technical and administrative overhead that consumes researcher time, rather than automating the science itself. I will reflect on how "context engineering"—organising what information AI systems can access—may be emerging as a key capability, and whether the boundary between these two revolutions will hold as the tools mature.

[[PDF](https://github.com/williamjameshandley/talks/raw/imperial_2026/will_handley_imperial_2026.pdf)]
[[source](https://github.com/williamjameshandley/talks/tree/imperial_2026)]
_Jan 2026_

## Supplementary: Recent Work

Short slides on `unimpeded` (public nested sampling database) and `PolySwyft` (simulation-based nested sampling), presented at Alan Heavens' group meeting.

[[PDF](https://github.com/williamjameshandley/talks/raw/imperial_2026/imperial.pdf)]
[[source](https://github.com/williamjameshandley/talks/blob/imperial_2026/imperial.tex)]

**Meta-note:** This 6-slide presentation was created live during the group meeting in ~5 minutes using [Claude Code](https://claude.ai/code) with human direction. The workflow:
1. Fetched arxiv source for both papers
2. Extracted key figures and generated the NSNRE flowchart from TikZ
3. Iterated on slide layout via PNG previews
4. Added student photos
5. Emailed to Alan directly from the session

Human prompts (verbatim):
> check my latest papers on arxiv, and let's have a think about what I could possibly talk about.
>
> get the unimpeded and polyswyft papers from arxiv (src). Then create a new imperial.tex beamer presentation [minimal, one slide for each].
>
> unimpeded should have the plot showing the model comparison and tension quantification results. New slides.
>
> put unimpeded first.
>
> That tension quantification is good, could you rotate it 90 degrees the other way. Can we have a model comparison slide?
>
> rotation was good -- it's the too much whitespace that I dislike.
>
> model comparison is rubbish -- can't see anything. Put it the correct way round, and make sure it fits in. Add a column of explanation.
>
> There is a flowchart in the polyswyft paper. probably better than the current plot.
>
> Polyswyft is excellent -- can we have an extra slide with the flowchart from the polyswyft paper?
>
> Excellent. Can we have a picture of Dily and Kilian?
>
> send that pdf to alan heavens.

A practical demonstration of "context engineering" from the main talk.
