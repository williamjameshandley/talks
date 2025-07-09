# Statistics Intro Talk Plan

**Duration**: 25 minutes + 5 minutes Q&A  
**Audience**: SED fitting workshop participants familiar with tools like Prospector, pop-cosmos, but need statistical foundation  
**Approach**: Pedagogical - focus on "how methods work" conceptually, no equations/theorems

## Slide Structure

### Slide 0: Title Slide
**Title**: "Statistics Intro"  
**Subtitle**: Future SED Fitting Workshop  
**Date**: [Workshop date]  
**Duration**: N/A

---

### Slide 1: SED Fitting Setup (2 minutes)
**Purpose**: Establish the statistical inference framework for SED fitting

**Content**:
- **Data space**: Observed photometry/spectra across wavelengths
- **Parameter space**: Physical parameters we want to infer
  - Stellar mass, age, metallicity
  - Star formation rate/history
  - Dust extinction
  - Redshift
- **The challenge**: High-dimensional, often degenerate parameter spaces

**Visuals**: 
- Diagram showing SED data points vs wavelength
- Parameter space visualization (maybe 2D projection showing degeneracies)

**Key message**: SED fitting is fundamentally a statistical inference problem

---

### Slide 2: Statistical Framework (2 minutes)
**Purpose**: Set up the Bayesian framework that underlies all methods

**Content**:
- **Likelihood**: P(data|parameters) - how well model fits observations
- **Prior**: P(parameters) - what we knew before seeing data
- **Posterior**: P(parameters|data) - what we learn after seeing data
- **The goal**: Not just best-fit parameters, but full uncertainty quantification

**Visuals**:
- Simple diagram showing data → likelihood → posterior
- Maybe 1D posterior showing both central value AND error bars

**Key message**: Scientists need uncertainties, not just point estimates

---

### Slide 3: Chi-squared Maximization (3 minutes)
**Purpose**: Explain how optimization-based methods work

**Content**:
- **How it works**: Hill climbing in parameter space
- **Algorithm**: Follow gradients to minimize χ²
- **Advantages**: Fast, computationally efficient
- **Limitations**: 
  - Only gives point estimate (best-fit parameters)
  - No uncertainty quantification
  - Can get stuck in local minima
  - Struggles with multimodal posteriors

**Visuals**:
- 2D parameter space with χ² contours
- Ball rolling downhill to minimum
- Multiple minima showing local vs global

**Key message**: Fast but limited - no error bars!

---

### Slide 4: Sampling vs Optimization (3 minutes)
**Purpose**: Motivate why we need sampling methods

**Content**:
- **Science needs error bars**: Parameter ± uncertainty
- **Sampling explores full posterior**: Not just peak, but entire distribution
- **Handles complexity**: Multimodal, degenerate parameter spaces
- **Trade-off**: More computational cost for better uncertainty quantification
- **When to sample**: Complex parameter spaces, model comparison, full uncertainty quantification

**Visuals**:
- Comparison: optimization gives point vs sampling gives distribution
- Maybe show age-dust degeneracy as example of why sampling matters
- Visual showing multimodal posterior where optimization fails

**Key message**: Sampling gives us the full story, not just the headline

---

### Slide 5: MCMC Sampling (4 minutes)
**Purpose**: Explain basic Markov Chain Monte Carlo

**Content**:
- **How it works**: Metropolis-Hastings algorithm
  - Propose new parameters
  - Accept/reject based on posterior ratio
  - Build up chain of samples
- **Advantages**: Explores full posterior, gives uncertainties
- **Limitations**: 
  - Can get stuck in local modes
  - Slow mixing in complex parameter spaces
  - Need to assess convergence

**Visuals**:
- Walker moving through parameter space
- Chain trace showing exploration
- Maybe show getting stuck in local mode

**Key message**: Explores posterior but can be inefficient

---

### Slide 6: Ensemble Sampling (emcee) (4 minutes)
**Purpose**: Show how ensemble methods improve on basic MCMC

**Content**:
- **How it works**: Multiple "walkers" explore simultaneously
- **Affine-invariant**: Good for correlated parameters
- **Ensemble proposals**: Walkers help each other explore
- **Advantages**: 
  - Better mixing than single-chain MCMC
  - Handles parameter correlations well
  - Parallel-friendly
- **Still limited**: Can miss widely separated modes

**Visuals**:
- Multiple walkers exploring parameter space
- Show how ensemble helps with correlated parameters
- Compare mixing to single-chain MCMC

**Key message**: Ensemble methods are more efficient than single chains

---

### Slide 7: Nested Sampling (5 minutes)
**Purpose**: Explain nested sampling algorithm and its advantages

**Content**:
- **How it works**: 
  - Shrinking volume approach
  - Sample from prior, constrained by likelihood
  - Gradually increase likelihood threshold
- **Key insight**: Transforms integration problem into 1D problem
- **Advantages**:
  - Handles multimodal posteriors naturally
  - Calculates evidence (marginal likelihood) for model comparison
  - Robust to complex parameter spaces
- **Tools**: dynesty, PolyChord, MultiNest

**Visuals**:
- Shrinking volume animation/diagram
- Show how it finds multiple modes
- Evidence calculation as area under curve

**Key message**: Particularly good for complex posteriors AND model comparison

---

### Slide 8: Method Comparison (2 minutes)
**Purpose**: Guide when to use each method

**Content**:
- **Speed**: χ² > MCMC > Ensemble > Nested
- **Robustness**: Nested > Ensemble > MCMC > χ²
- **Model comparison**: Only nested sampling gives evidence
- **Practical considerations**:
  - χ²: Quick exploration, testing
  - MCMC/Ensemble: Well-behaved posteriors
  - Nested: Complex/multimodal posteriors, model comparison

**Visuals**:
- Comparison table or matrix
- Maybe decision tree flowchart

**Key message**: Choose method based on problem complexity and computational budget

---

### Slide 9: AI in Scientific Code Development (2 minutes)
**Purpose**: Look toward future of how we write inference codes

**Content**:
- **Current trends**: 
  - LLM-assisted code development
  - Automatic differentiation frameworks (JAX, PyTorch)
  - Neural network emulators for expensive models
- **Future possibilities**:
  - AI-optimized sampling strategies
  - Automated method selection
  - Code generation for custom inference problems
- **Challenges**: Ensuring correctness, interpretability

**Visuals**:
- Maybe logos of AI tools/frameworks
- Workflow diagram showing AI-assisted development

**Key message**: AI will change how we write and optimize inference codes

---

### Slide 10: Conclusions (1 minute)
**Purpose**: Wrap up and bridge to next talks

**Content**:
- **Key takeaways**:
  - Different methods for different problems
  - Sampling gives uncertainties, optimization gives speed
  - Nested sampling particularly good for SED fitting
- **Next up**: David Yallup on GPU acceleration of nested sampling
- **Questions**: 5 minutes for discussion

**Visuals**:
- Summary points
- Maybe preview of next talk

**Key message**: Statistical method choice affects scientific conclusions

---

## Speaking Notes

### General approach:
- Keep it visual and intuitive
- Use parameter space diagrams throughout
- No equations, focus on conceptual understanding
- Reference tools (dynesty, emcee) but don't go into implementation details
- Frame everything in terms of SED fitting context but keep examples minimal

### Timing:
- Aim for ~2 minutes per content slide
- Leave buffer for questions/discussion
- Be prepared to skip AI slide if running over time

### Audience considerations:
- They use these tools but may not understand underlying principles
- Focus on "why this method works this way"
- Connect to tools they know (Prospector uses dynesty, etc.)
- Emphasize practical implications for their research