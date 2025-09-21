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

```python
>>> from ultrasphere_harmonics import harmonics
>>> harmonics(1)
2
```

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

| phase      | definition                                                                                                           | difference from `Phase(0)`   |
| ---------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `Phase(0)` | $Y_n^{(0),m} = \sqrt{\frac{2n+1}{4\pi} \frac{(n-\|m\|)!}{(n+\|m\|)!}} P_n^{\|m\|} (\cos \theta) e^{i m \phi}$        | $1$                          |
| `Phase(1)` | $Y_n^{(1),m} = \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}} P_n^m (\cos \theta) e^{i m \phi}$                      | $(-1)^{\frac{\|m\| - m}{2}}$ |
| `Phase(2)` | $Y_n^{(2),m} = (-1)^m \sqrt{\frac{2n+1}{4\pi} \frac{(n-\|m\|)!}{(n+\|m\|)!}} P_n^{\|m\|} (\cos \theta) e^{i m \phi}$ | $(-1)^m$                     |
| `Phase(3)` | $Y_n^{(3),m} = (-1)^m \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}} P_n^m (\cos \theta) e^{i m \phi}$               | $(-1)^{\frac{\|m\| + m}{2}}$ |

Note that $\forall m \in \mathbb{Z}. (-1)^m = (-1)^{-m}$ holds.

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
