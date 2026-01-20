# Imperial Talk Draft - Gemini v1

## Slide 1: Two Revolutions: GPUs for Science, AI for Scientists

**Will Handley**
*Cavendish Laboratory, University of Cambridge*

- **The Premise**: We are living through two simultaneous technological shifts.
- **Revolution 1**: GPU hardware, driven by the AI boom, allows us to scale *classical* statistical methods (like Nested Sampling) by factors of 100–1000×.
- **Revolution 2**: Large Language Models (AI) are changing how we write code and manage research, potentially solving the "code debt" crisis in cosmology.
- **The Argument**: These are distinct. One accelerates the *calculation*; the other accelerates the *researcher*.

[Figure: A split screen image. Left: A complex posterior distribution (triangle plot). Right: A terminal window showing AI-assisted code generation.]

---

## Slide 2: Inference Across Astrophysics

**The same mathematical problem appears in three different guises**

- **Cosmology**:
  - *Parameter Estimation*: Hubble constant ($H_0$), Matter density ($\Omega_m$).
  - *Model Comparison*: $\Lambda$CDM vs $w$CDM vs $w_0w_a$CDM (Is Dark Energy evolving?).
- **Gravitational Waves**:
  - *Parameter Estimation*: Masses, spins, sky localization.
  - *Model Comparison*: NS-NS vs BH-BH events.
- **Exoplanets**:
  - *Parameter Estimation*: Orbital periods, radii, atmospheric composition.
  - *Model Comparison*: How many planets are in this system? (1 vs 2 vs 3...).

[Figure: Three panels side-by-side. 1. DESI posterior contours. 2. GW sky localization map. 3. Exoplanet transit light curve with model fits.]

---

## Slide 3: The Common Challenge

**Why standard optimization ($\chi^2$) fails us**

- **High-Dimensional**:
  - Cosmo: 6+ base parameters + dozens of nuisance parameters.
  - GW: 15+ parameters.
  - Volume grows exponentially with dimension.
- **Degenerate**:
  - Parameters are strongly correlated (e.g., the $H_0 - \Omega_m$ "banana").
- **Multimodal**:
  - Multiple distinct solutions (e.g., inclination degeneracies in GWs).
- **The Consequence**: We cannot rely on point estimates. We need full Bayesian posterior sampling.

[Figure: A complex 2D posterior showing "banana" shapes and multiple isolated modes, perhaps the Himmelblau function or a specific cosmological tension plot.]

---

## Slide 4: Why Nested Sampling?

**Beyond MCMC for rigorous science**

- **The Problem with MCMC**: Standard Metropolis-Hastings or HMC can get stuck in local modes and struggle with phase transitions.
- **Nested Sampling (Skilling 2004)**:
  - Evolves a cloud of "live points" that contract around the peak likelihood.
  - **Key Feature 1**: Handles multimodality naturally (global exploration).
  - **Key Feature 2**: Computes the **Bayesian Evidence** ($\mathcal{Z}$) as a primary output.
    - Essential for Model Comparison: $\mathcal{P}(M|D) \propto \mathcal{Z}(D|M)\mathcal{P}(M)$.
    - Occam's Razor is built-in.

[Figure: Animation or schematic of Nested Sampling: points compressing into the high-likelihood region over time.]

---

## Slide 5: The Computational Bottleneck

**The cost of rigour**

- Nested Sampling is powerful, but computationally expensive.
- Requires many likelihood evaluations to shrink the prior volume.
- **Traditional CPU approach**:
  - Few powerful cores ($\sim$10s).
  - Sequential evaluation of live points.
- **The Hardware Shift**:
  - Modern HPC is dominated by GPUs (driven by Deep Learning demand).
  - GPUs: Thousands of simple cores, massive parallelism.
- **The Opportunity**: Nested Sampling is "embarrassingly parallel" if implemented correctly.

[Figure: Bar chart comparing CPU vs GPU core counts and memory bandwidth.]

---

## Slide 6: Two Independent Capabilities

**Clarifying a common misconception in scientific computing**

1.  **Differentiable Programming (The "Gradient" Revolution)**
    - Tools: JAX, PyTorch.
    - Capability: Automatic differentiation ($\nabla_\theta \log \mathcal{L}$).
    - Enables: HMC, NUTS, Gradient Descent.
2.  **GPU Parallelization (The "Vector" Revolution)**
    - Capability: Running 1000s of calculations simultaneously.
    - Enables: Vectorized likelihoods, massive ensemble sampling.

- **Key Insight**: You can have #2 without #1.
  - We can use GPUs to accelerate Nested Sampling *without* needing gradients.
  - This is crucial for "black box" likelihoods common in astrophysics.

[Figure: Diagram separating "Gradient-based methods (HMC)" from "Population-based methods (NS/Ensemble)" and mapping both to GPU hardware.]

---

## Slide 7: BlackJAX: GPU Native Sampling

**A unified framework for accelerated inference**

- **The Solution**: Move the entire sampling loop to the GPU.
  - Avoid expensive CPU $\leftrightarrow$ GPU data transfer.
- **BlackJAX**: A library of composable inference building blocks in JAX.
  - Supports: MCMC, NUTS, SMC, and now **Nested Sampling**.
- **Performance**:
  - We are seeing speedups of **100–1000×** compared to traditional CPU implementations (like `PolyChord` or `dynesty`).
  - Days of compute become minutes.

[Figure: Screenshot of BlackJAX code structure or logo, plus a performance scaling plot.]

---

## Slide 8: Case Study: Cosmology (CMB & Shear)

**Toby Lovick (PhD Student)**

- **Science Case**: Cosmic Shear and CMB analysis.
- **Challenge**: 37+ parameters. Comparing $\Lambda$CDM vs $w_0w_a$CDM.
- **Method**: JAX-based neural emulators for the theory code + GPU Nested Sampling.
- **Result**:
  - **300× speedup** vs CPU PolyChord.
  - **10× speedup** vs GPU NUTS (HMC).
- **Impact**: We can now calculate the Bayesian Evidence for complex shear models in minutes, enabling routine rigorous model comparison.

[Figure: Plot showing posterior contours for Cosmic Shear parameters, comparing CPU vs GPU runtimes.]

---

## Slide 9: Case Study: Type Ia Supernovae

**Sam Leeney (PhD Student)**

- **Science Case**: Supernova cosmology for LSST.
- **Challenge**: Contamination from non-Ia supernovae. Manual rejection doesn't scale to LSST volumes.
- **Solution**: **Bayesian Anomaly Detection**.
  - Model the probability of contamination *per object* within the fit.
  - Requires massive sampling resources to integrate over contamination probabilities.
- **Result**: Automatic identification of outliers/corrupted bands without human intervention.
- **Finding**: Contaminants are systematically brighter/bluer; ignoring this biases cosmology.

[Figure: Light curves of Type Ia Supernovae showing outliers identified by the GPU-accelerated anomaly detection.]

---

## Slide 10: Case Study: Gravitational Waves

**Metha Prathaban (PhD Student)**

- **Science Case**: Parameter estimation for Binary Black Holes (BBH).
- **Tool**: `bilby` is the standard, but slow on CPUs.
- **Work**: Ported the acceptance-walk Nested Sampling algorithm to JAX/BlackJAX.
- **Performance**:
  - **20–40× speedup** for BBH events.
  - Single GPU replaces a CPU cluster node.
- **Validation**: Produces identical posteriors and evidences to the standard CPU implementation.

[Figure: "Corner plot" comparison showing perfect overlap between CPU `bilby` and GPU `blackjax` posteriors.]

---

## Slide 11: The Second Revolution: AI for Scientists

**Shifting gears: How did we build this code?**

- Writing high-performance GPU code (CUDA, JAX) is historically difficult.
- **The "80/20" Rule of Science**:
  - 20% of time: "Hard thinking" (Physics, insight).
  - 80% of time: "Boring tasks" (Coding, debugging, formatting, data munging).
- **The Role of AI**:
  - Large Language Models (LLMs) are not about "automating science" (the 20%).
  - They are about collapsing the 80% of overhead.
- **Example**: Much of the BlackJAX Nested Sampling implementation was "vibe-coded" (interactive development) with LLMs.

[Figure: Pie chart of "Time spent by a PhD student" showing the large "Technical Overhead" slice shrinking with AI.]

---

## Slide 12: Three Ways to Use AI

**Finding the sweet spot for research**

1.  **Foundation Models**: "Chatbots." Good for brainstorming, bad for complex implementation.
2.  **Autonomous Agents**: "Let it run overnight."
    - Generates volume, not quality.
    - *Risk*: Hallucinated science, zero accountability.
3.  **Supervised Execution (The Sweet Spot)**:
    - Human provides **Intent** + **Verification**.
    - AI provides **Implementation**.
    - The human stays "in the loop" but operates at a higher level of abstraction.

[Figure: Diagram showing the three levels, highlighting "Supervised Execution" as the preferred path.]

---

## Slide 13: Generation, Selection, Propagation

**An evolutionary framework for AI-assisted research**

- **Generation**: AI excels here.
  - Writing boilerplate JAX code, drafting documentation, creating GUIs.
- **Selection**: AI struggles here.
  - *Taste*: Knowing which idea is physically sound.
  - *Truth*: Verifying the math is correct.
  - **This is the scientist's new primary job.**
- **Propagation**:
  - Getting the idea into the community (papers, talks).
  - AI helps with formatting, but the *institutional trust* comes from the human.

[Figure: Cycle of Generation -> Selection -> Propagation, marking where AI vs Human excels.]

---

## Slide 14: From Prompt Engineering to Context Engineering

**The skill of 2026**

- **2024/25**: "Prompt Engineering" (How to ask the question).
- **2026**: "Context Engineering" (Organising what the model knows).
- **The Bottleneck**: Models are smart (Claude Opus 4.5, GPT-5 class), but they are *stateless*.
- **The Solution**:
  - **Skills**: Read-only text files defining best practices (e.g., `physics_conventions.md`).
  - **MCPs (Model Context Protocols)**: Tools that allow the AI to read/write files, run code, and query databases.

[Figure: Schematic of an LLM connected to a "Context Window" which is fed by specific files and tools.]

---

## Slide 15: Case Study: Cosmosis to Cobaya

**Building a verified likelihood tool**

- **Task**: Port a legacy cosmology likelihood from `Cosmosis` to `Cobaya`.
- **The AI Workflow**:
  - AI drafts the code.
  - **Human constraint**: "Must match to floating-point precision."
  - AI: "It matches to 5 decimal places."
  - Human: "Not good enough. Find the bug."
- **Outcome**: The AI found subtle indexing errors that a human might miss, but only because the *human set the rigorous standard*.
- **Lesson**: AI allows us to enforce higher standards of code quality because fixing bugs is cheaper.

[Figure: Code diff or a terminal screenshot showing a test passing with high precision.]

---

## Slide 16: Case Study: Jim Emulators

**James Alvey & Will Handley**

- **Goal**: Build a JAX-based emulator infrastructure for Gravitational Waves.
- **Process**:
  - Started at 9:00 AM with nothing.
  - Used AI to generate boilerplate, tests, and documentation.
  - Human focused on the physical architecture and validation.
  - By 6:00 PM: A fully working, documented, tested package.
- **The Feeling**: "We didn't feel tired. We felt invigorated."
- We spent the day doing *physics*, not fighting syntax.

[Figure: Screenshot of the `jim` repository or a flow chart of the emulator logic.]

---

## Slide 17: GTD: Giving AI Memory

**Solving the stateless problem**

- **GTD (Getting Things Done)**: A methodology for organizing tasks.
- **The Implementation**: A system (built with AI assistance in Rust) that maintains the state of a project.
- **How it works**:
  - The AI reads the current "State" (files, logs).
  - It proposes the next "Action".
  - It records the "Decision".
- This allows AI to work on projects over weeks, not just in a single chat window.

[Figure: Diagram of the GTD loop: Capture -> Clarify -> Organize -> Reflect -> Engage.]

---

## Slide 18: Context Engineering for Hardware

**Case Study: The Water-Damaged Laptop**

- **Situation**: Laptop water damage. Multiple component failures.
- **The AI Role**: Diagnostic partner.
  - AI: "Given symptoms X and Y, check component Z."
  - Human: "Component Z is fine, but I see corrosion on W."
  - AI: "Refining hypothesis: It could be a short in the power button circuit."
- **Result**: Fixed.
- **Why it worked**: The AI had access to technical manuals (Context) and logical reasoning; the Human provided eyes and hands (Selection/Action).

[Figure: Photo of the disassembled laptop or the diagnostic chat log.]

---

## Slide 19: The Vision: Solving "Code Debt"

**Cosmology is riven with technical debt**

- PhD students spend years learning to wrestle with `Cobaya`, `Cosmosis`, or `MontePython`.
- These are massive, complex codes.
- **The Future**:
  - "Sophisticated Autocomplete" becomes "Sophisticated Architecture."
  - Instead of maintaining legacy monoliths, we might move to smaller, modular scripts generated on-demand.
  - **Skills** (text descriptions of physics) become the source of truth, not the Fortran code written in 1998.

[Figure: Image representing a transition from a "Monolithic Codebase" to "Modular, AI-assembled components."]

---

## Slide 20: Conclusion: The Boundary

**Will these two revolutions remain separate?**

- Currently:
  - **GPUs** are for the *Likelihood* (The Science).
  - **AI** is for the *Workflow* (The Scientist).
- **Future**:
  - As we move to Neural Posterior Estimation (NPE) and emulator-based inference, the boundary blurs.
  - But the *role of the human* remains distinct:
    - We are the **Selectors**.
    - We define the **Context**.
    - We provide the **Trust**.

[Figure: A Venn diagram of "GPU Compute" and "AI Workflow" with "The Scientist" at the intersection.]

---

## Slide 21: Summary

**Two Revolutions: GPUs for Science, AI for Scientists**

1.  **GPU Revolution**:
    - Nested Sampling on GPUs is here (BlackJAX).
    - It enables rigorous Bayesian evidence calculation at scale.
    - Independent of gradients: works for complex, discrete astrophysics.
2.  **AI Revolution**:
    - Don't use AI to write your paper; use it to write your tools.
    - **Context Engineering** is the key skill: Manage your information, not just your prompts.
    - Allows us to tackle "Code Debt" and focus on physics.

**Resources**:
- Nested Sampling Book: `handley-lab.co.uk/nested-sampling-book`
- Marketplace of Skills: `github.com/fundamental-physics/marketplace`

[Figure: QR codes to the resources.]
