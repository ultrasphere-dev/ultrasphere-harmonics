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

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install ultrasphere-harmonics
```

## How it works

Let

$$
\begin{aligned}
P_n (\mathbb{R}^d) &:= \left\{\sum_{|\alpha| = n - 1} c_\alpha x^\alpha | c_\alpha \in \mathbb{C}\right\} \\
H_n (\mathbb{R}^d) &:= \left\{f \in P_n (\mathbb{R}^d) | \Delta f = 0\right\} \\
H_n (\mathbb{S}^{d-1}) &:= \left\{f|_{\mathbb{S}^{d-1}} | f \in H_n (\mathbb{R}^d)\right\}
\end{aligned}
$$

spaces of homogeneous polynomials, harmonic polynomials, and spherical harmonics, respectively.

We are interested to compute orthonormal basis of $H_n (\mathbb{S}^{d-1})$ which coressponds to one of spherical coordinates. Any element $Y^l$ in such basis can be expressed as product of eigenfunctions $\psi^l_\Theta$ with different quantum numbers $l$:

$$
Y^l(\theta) = \prod_{\Theta \in \text{Nodes}} \psi^l_\Theta (\theta_{\Theta})
$$

Each $\psi^l_\Theta$ only depends to 1,2 or 3 quantum number depending on the type of $\Theta$.

To compute $Y$, this package do the following:

1. Compute all $\psi_\Theta$ and put them to a `Mapping[TSpherical, Array]`, where the dimension of each array is minimal (1, 2, or 3)
2. Expand the dimension of each array to $d-1$ and reorder them, so that they are broadcastable
3. Multiply all arrays
4. Remove disallowed combinations of quantum numbers

Example for Type **ba** coordinates, the well-known spherical harmonics:

1. Compute $\exp(i m \phi)/\sqrt{2 \pi}$ and $P_n^m (\cos \theta)$, where first array is of shape `(2n-1, )` and second array is of shape `(2n-1, n)`, and $P_n^m$ is the normalized (in terms of $L^2$) associated Legendre polynomial.
2. Expand the first array to shape `(2n-1, 1)`. Second array (shape `(2n-1, n)`) does not need to be expanded. Now they are broadcastable.
3. Multiply them to get an array of shape `(2n-1, n)`
4. Remove disallowed combinations of quantum numbers, $|m| > n$, to get an array of shape `(n^2, )`

Note that this package supports **any** spherical coordinates.

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
