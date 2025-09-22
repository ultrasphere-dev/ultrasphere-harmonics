# ultrasphere-harmonics

<p align="center">
  <a href="https://github.com/ultrasphere-dev/ultrasphere-harmonics/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/ultrasphere-dev/ultrasphere-harmonics/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://ultrasphere-harmonics.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/ultrasphere-harmonics.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/ultrasphere-dev/ultrasphere-harmonics">
    <img src="https://img.shields.io/codecov/c/github/ultrasphere-dev/ultrasphere-harmonics.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/ultrasphere-harmonics/">
    <img src="https://img.shields.io/pypi/v/ultrasphere-harmonics.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/ultrasphere-harmonics.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/ultrasphere-harmonics.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://ultrasphere-harmonics.readthedocs.io" target="_blank">https://ultrasphere-harmonics.readthedocs.io </a>

**Source Code**: <a href="https://github.com/ultrasphere-dev/ultrasphere-harmonics" target="_blank">https://github.com/ultrasphere-dev/ultrasphere-harmonics </a>

---

Hyperspherical harmonics in NumPy / PyTorch / JAX

![Spherical Harmonics Expansion of Stanford Bunny](https://raw.githubusercontent.com/ultrasphere-dev/ultrasphere-harmonics/main/expand_bunny.gif)

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install ultrasphere-harmonics
```

## Usage

All functions support batch calculation.

Following [Generalized universal function API (NumPy)](https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html),
**Batch** dimensions are mapped to the **first** dimensions of arrays and **Core** dimensions are mapped to the **last** dimensions of arrays.

### Spherical Harmonics

```python
>>> from array_api_compat import numpy as np
>>> from ultrasphere import create_spherical
>>> from ultrasphere_harmonics import harmonics
>>> harm = harmonics( # flattened output
...     create_spherical(), # structure for spherical coordinates
...     {"theta": np.asarray(0.5), "phi": np.asarray(1.0)}, # spherical coordinates
...     n_end=2, # maximum degree - 1
...     phase=0 # the phase convention (see below for details)
... )
>>> np.round(harm, 2)
array([0.28+0.j  , 0.43+0.j  , 0.09+0.14j, 0.09-0.14j])
```

#### Batch calculation

```python
>>> harm = harmonics( # flattened output
...     create_spherical(), # structure for spherical coordinates
...     {"theta": np.asarray([0.5, 0.6]), "phi": np.asarray([1.0])}, # spherical coordinates
...     n_end=2, # maximum degree - 1
...     phase=0 # the phase convention (see below for details)
... )
>>> np.round(harm, 2)
array([[0.28+0.j  , 0.43+0.j  , 0.09+0.14j, 0.09-0.14j],
       [0.28+0.j  , 0.4 +0.j  , 0.11+0.16j, 0.11-0.16j]])
```

### Hyperspherical Harmonics

Spherical harmonics for higher dimensions or lower dimensions can be calculated by changing the structure of spherical coordinates.

```python
>>> from ultrasphere import create_standard
>>> harm = harmonics( # flattened output
...     create_standard(4), # structure for spherical coordinates
...     {"theta0": np.asarray(0.5), "theta1": np.asarray(0.75), "theta2": np.asarray(1.0), "theta3": np.asarray(1.25)}, # spherical coordinates
...     n_end=2, # maximum degree - 1
...     phase=0 # the phase convention (see below for details)
... )
>>> np.round(harm, 2)
array([0.19+0.j  , 0.38+0.j  , 0.15+0.j  , 0.08+0.j  , 0.03+0.08j,
       0.03-0.08j])
```

### Expansion and Evaluation

#### Expansion

```python
>>> from ultrasphere_harmonics import expand, expand_evaluate
>>> def my_function(spherical):
...     return np.sin(spherical["theta"]) + np.cos(spherical["phi"])
>>> expansion_coef = expand(
...     create_spherical(),
...     my_function,
...     n_end=6, # maximum degree - 1
...     n=12, # number of points for numerical integration
...     does_f_support_separation_of_variables=False,
...     phase=0,
...     xp=np
... )
>>> np.round(expansion_coef, 2)
array([ 2.78+0.j, -0.  +0.j,  1.71+0.j,  1.71-0.j, -0.78+0.j, -0.  +0.j,
       -0.  +0.j, -0.  -0.j, -0.  -0.j, -0.  +0.j,  0.4 +0.j,  0.  +0.j,
       -0.  +0.j, -0.  -0.j,  0.  -0.j,  0.4 -0.j, -0.13+0.j, -0.  -0.j,
       -0.  +0.j,  0.  +0.j, -0.  +0.j, -0.  -0.j,  0.  -0.j, -0.  -0.j,
       -0.  +0.j,  0.  +0.j,  0.2 +0.j, -0.  -0.j, -0.  +0.j, -0.  +0.j,
       -0.  -0.j, -0.  +0.j, -0.  -0.j, -0.  -0.j, -0.  +0.j,  0.2 -0.j])
```

#### Evaluation

```python
>>> spherical = {
...     "theta": np.asarray(0.5),
...     "phi": np.asarray(0.75),
... }
>>> my_function_expected = my_function(spherical)
>>> np.round(my_function_expected, 6)
np.float64(1.211114)
>>> my_function_approx = expand_evaluate(
...     create_spherical(),
...     expansion_coef,
...     spherical,
...     phase=0
... )
>>> np.round(my_function_approx, 6)
np.complex128(1.248959+0j)
```

See [API Reference](https://ultrasphere-harmonics.readthedocs.io/en/latest/ultrasphere_harmonics.html) for further details and examples.

## 2 formats for storing spherical harmonics

| Format           | indexing `(n, m)` in type **ba** coordinates   |
| ---------------- | ---------------------------------------------- | ---------------- |
| Multidimensional | `array[..., n, m]`                             |
| Flatten          | `array[..., (n - 1) ** 2 + (m % (2 * n - 1))]` | Memory efficient |

### Advantages of Multidimensional format

- Easy to understand and use
- Easy to slice
- Summantion over specific quantum number is easy

### Advantages of Flatten format

- Memory efficient
- Easy to use linear algebra functions like `np.linalg.solve`, `np.inv` etc.
- Summation over all quantum numbers is easy

The format can be converted using `flatten_harmonics()` and `unflatten_harmonics()` functions.

## 4 different definitions of spherical harmonics for type **ba** coordinates

There are 4 different definitions of spherical harmonics in the literature. This package supports all of them by changing `phase` parameter.

### Associated Legendre Polynomial

The associated Legendre polynomial is defined as follows:

$$
P_n^m (x) = (1 - x^2)^{\frac{m}{2}} \frac{d^m}{dx^m} P_n (x)
$$

In some literature, the Condon-Shortley phase $(-1)^m$ is included in the definition of $P_n^m$ as follows:

$$
{P'}_n^m (x) = (-1)^m (1 - x^2)^{\frac{m}{2}} \frac{d^m}{dx^m} P_n (x)
$$

### Spherical Harmonics

| phase      | definition                                                                                                           | difference from `Phase(0)`   | Used in           |
| ---------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------- |
| `Phase(0)` | $Y_n^{(0),m} = \sqrt{\frac{2n+1}{4\pi} \frac{(n-\|m\|)!}{(n+\|m\|)!}} P_n^{\|m\|} (\cos \theta) e^{i m \phi}$        | $1$                          | [Kress2014]       |
| `Phase(1)` | $Y_n^{(1),m} = \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}} P_n^m (\cos \theta) e^{i m \phi}$                      | $(-1)^{\frac{\|m\| - m}{2}}$ | Wolfram MathWorld |
| `Phase(2)` | $Y_n^{(2),m} = (-1)^m \sqrt{\frac{2n+1}{4\pi} \frac{(n-\|m\|)!}{(n+\|m\|)!}} P_n^{\|m\|} (\cos \theta) e^{i m \phi}$ | $(-1)^m$                     | [Gumerov2004]     |
| `Phase(3)` | $Y_n^{(3),m} = (-1)^m \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}} P_n^m (\cos \theta) e^{i m \phi}$               | $(-1)^{\frac{\|m\| + m}{2}}$ | Scipy             |

Note that $\forall m \in \mathbb{Z}. (-1)^m = (-1)^{-m}$ holds.

- [Gumerov2004]: Gumerov, N. A., & Duraiswami, R. (2004). Recursions for the Computation of Multipole Translation and Rotation Coefficients for the 3-D Helmholtz Equation. SIAM Journal on Scientific Computing, 25(4), 1344–1381. https://doi.org/10.1137/S1064827501399705
- [Kress2014]: Kress, R. (2014). Linear Integral Equations (Vol. 82). Springer New York. https://doi.org/10.1007/978-1-4614-9593-2

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
