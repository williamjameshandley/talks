# GPU Accelerated Nested Sampling for 21-cm Cosmology

Nested Sampling is a powerful Monte Carlo method for robust parameter estimation and model comparison in high-dimensional, complex likelihood landscapes. Traditionally implemented on CPUs, its use in many communities—including those working on 21-cm cosmology—has sometimes been limited by legacy code structures. In this work, we reformulate Nested Sampling to fully exploit modern GPU hardware, leveraging vectorization to dramatically accelerate numerical inference while maintaining accuracy in evidence evaluation.

Our public implementation is designed not only to provide state-of-the-art performance improvements, but also to address key challenges in 21-cm cosmology. In this context, GPU-accelerated Nested Sampling shows particular promise for:
- **Beam Modelling:** Optimizing the characterization and variation of instrumental beams.
- **Hyperfine Calibration of Radiometers:** Enhancing the precision and reliability of radiometric calibration essential for 21-cm experiments.
- **Sky Modelling:** Improving the reconstruction of foregrounds and sky signals.
- **Theoretical Model Comparison:** Facilitating robust selection among competing astrophysical models.

Furthermore, the approach naturally complements GPU-accelerated emulators, which are increasingly popular in our community for rapid approximations of complex simulations.

[[PDF](https://github.com/williamjameshandley/talks/raw/caltech_2025/will_handley_caltech_2025.pdf)]