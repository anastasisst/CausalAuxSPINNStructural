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
demo. Increase `EPOCHS` in the hyperparameter cell to reach paper-quality
errors.

Dependencies: `jax` (CUDA), `flax`, `optax`, `numpy`, `matplotlib`,
`tqdm`.

## Citation

If you use this code, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
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
