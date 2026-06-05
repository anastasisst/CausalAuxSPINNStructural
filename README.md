# Causal Aux SPINN - Demo Notebooks

Self-contained JAX/Flax notebooks accompanying the paper. Each notebook
inlines its own SPINN model, autodiff helpers, data generator, loss, and
training loop - no external imports required beyond the standard JAX
stack.

## Benchmarks

| # | Notebook                              | Problem |
|---|---------------------------------------|---------|
| 1 | `01_EB_Causal_SPINN.ipynb`            | Euler-Bernoulli beam, 16th vibration mode |
| 2 | `02_EB_Causal_AUX_SPINN.ipynb`        | Euler-Bernoulli beam with auxiliary variable $\nu = u_{xx}$ |
| 3 | `03_Timoshenko_Causal_SPINN.ipynb`    | Timoshenko beam, 11th vibration mode (coupled $u$, $\theta$) |
| 4 | `04_Membrane_Causal_SPINN.ipynb`      | Simply-supported square membrane, 2D wave equation |
| 5 | `05_Plate_Causal_AUX_SPINN.ipynb`     | Kirchhoff-Love plate, biharmonic with auxiliary $\nu = \nabla^2 u$ |

## Quick start

Open any notebook and run all cells. Defaults are sized for a few-minute
demo.

Dependencies: `jax` (CUDA), `flax`, `optax`, `numpy`, `matplotlib`,
`tqdm`.

## Citation

If you use this code, please cite:

```bibtex
@article{STAMOU2026115285,
title = {A unified causality-enhanced separable physics-informed neural network for predicting beam and plate dynamics},
journal = {Engineering Applications of Artificial Intelligence},
volume = {181},
pages = {115285},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.115285},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626015691},
author = {Anastasios Stamou and Taniya Kapoor and Michalis Fragiadakis},
keywords = {Separable physics-informed neural networks, Causal training, Auxiliary variables, Structural dynamics, High-order partial differential equations, Inverse identification},
abstract = {A thorough understanding of the dynamic behavior of structural elements, such as beams, plates, and membranes, is crucial for reliable structural analysis. This behavior is typically governed by higher-order partial differential equations (PDEs), which accurately capture their spatiotemporal response. However, the presence of high-order differential operators and inherent high-frequency components, especially over large spatial domains, poses challenges for Physics-Informed Neural Networks (PINNs). To tackle this challenge, we propose a causality-informed Separable Physics-Informed Neural Network (SPINN) framework augmented with auxiliary variables (Causal Aux SPINN) to enhance the performance of PINNs in structural engineering applications. The proposed framework integrates three key components. First, the SPINN formulation decomposes the spatiotemporal solution into low-rank components, allowing efficient representation of high-dimensional dynamic fields with reduced computational cost. Second, the causality-informed loss function re-weights the training objective to respect the temporal evolution of structural dynamics and mitigate error accumulation over time. Third, the auxiliary-variable formulation introduces additional outputs for second-order fields, such as curvature-related quantities, enabling accurate enforcement of fourth-order differential operators without relying solely on successive differentiation of displacement fields. The proposed “Causal Aux SPINN” algorithm is evaluated on both forward and inverse benchmarks in structural dynamics, including Euler–Bernoulli and Timoshenko beams, membranes, and Kirchhoff–Love plates. Across all benchmarks, the Causal Aux SPINN method consistently reduces error and improves robustness, while maintaining computational efficiency comparable to the Baseline SPINN formulation. Overall, coupling causal weighting with auxiliary-variable learning is shown to provide an effective framework for structural engineering applications.}
}
```

## Acknowledgements

Built on the **Separable Physics-Informed Neural Networks (SPINN)**
architecture introduced by Cho et al. (NeurIPS 2023):

```bibtex
@article{cho2023separable,
  title   = {Separable Physics-Informed Neural Networks},
  author  = {Cho, Junwoo and Nam, Seungtae and Yang, Hyunmo and Yun, Seok-Bae
             and Hong, Youngjoon and Park, Eunbyung},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2023}
}
```

Original code: <https://github.com/stnamjef/SPINN>.
