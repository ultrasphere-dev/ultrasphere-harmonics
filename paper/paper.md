---
title: "ultrasphere and ultrasphere-harmonics: Python packages for Vilenkin–Kuznetsov–Smorodinsky polyspherical coordinates and hyperspherical harmonics methods in array API"
tags:
  - Python
authors:
  - given-names: Hiromochi
    surname: Itoh
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - given-names: Kei
    surname: Matsushima
    affiliation: 2
  - given-names: Takayuki
    surname: Yamada
    affiliation: 3
    equal-contrib: false
    corresponding: true
affiliations:
  - name: Department of Mechanical Engineering, Graduate School of Engineering, The University of Tokyo, Japan
    index: 1
    ror: 057zh3y96
  - name: Graduate School of Advanced Science and Engineering, Hiroshima University, Japan
    index: 2
    ror: 03t78wx29
  - name: Department of Strategic Studies, Institute of Engineering Innovation, Graduate School of Engineering, The University of Tokyo
    index: 3
    ror: 057zh3y96
date: 23 September 2025
bibliography: paper.bib
---

# Summary

Spherical harmonics, which are the solutions to the angular part of the 3 dimensional laplace equation, have been widely used in various fields of science and engineering.
While numerous software packages exist for standard three-dimensional spherical harmonics, many modern scientific challenges require working with hyperspherical harmonics, which are spherical harmonics in higher dimensions.
Hyperspherical harmonics have been widely used for various applications, including many-body problems in quantum mechanics [@fock_zur_1935], representation of crystallographic textures [@bonvallet_3d_2007], description of 3D models [@bonvallet_3d_2007], representation of brain structures [@hosseinbor_4d_2013], representation of the Head-Related Transfer Function, which characterizes how an ear receives a sound from a point in space [@szwajcowski_continuous_2023].
However, there is a barrier for researchers to implement spherical harmonics and hyperspherical harmonics methods in their work, as the implementation is often specific to the dimension and coordinate system used, requiring significant effort to adapt the code for different dimensions or coordinate systems.
To address this, we have developed a software package for implementing spherical harmonics techniques in arbitrary dimensions and coordinate systems.
Our packages would allow researchers to easily implement and extend their work to higher dimensions, for example, from 2D to 3D and further to 4D, without having to duplicate code for each dimension.

# Statement of need

`ultrasphere` is a Python package for Vilenkin–Kuznetsov–Smorodinsky (VKS) polyspherical coordinate systems [@vilenkin_representation_1993].
Built on top of `ultrasphere`, `ultrasphere-harmonics` implements hyperspherical harmonics methods for any type of polyspherical coordinates.
While spherical harmonics in 3D itself have been widely implemented in various software packages, such as Scipy [@2020SciPy-NMeth], hyperspherical harmonics are rarely implemented, and software packages which supports arbitrary VKS polyspherical coordinates are not known.
To remedy this, our packages allow to convert between Cartesian coordinates and VKS polyspherical coordinates, compute hyperspherical harmonics, elementary solutions to the Helmholtz equation, hyperspherical expansion of a function, and the translational coefficients of elementary solutions of the Helmholtz equation under arbitrary VKS polyspherical coordinates and dimensions.
The underlying implementation leverages the "method of trees" [@vilenkin_representation_1993; @cohl_fourier_2012], the rooted tree representation of VKS coordinates with the help of NetworkX [@hagberg_exploring_2008].
A command-line application that solves for acoustic scattering from a sound-soft sphere using arbitrary VKS polyspherical coordinates is included to illustrate a practical use case.

Spherical expansion methods are sometimes computationally expensive, especially in higher dimensions.
To utilize modern high-performance computing resources, which environment is recently diversified, our api is made to be compatible with the array API standard [@meurer_python_2023], ensuring that the same code can run on multiple array libraries (e.g., NumPy [@harris_array_2020], PyTorch [@paszke_pytorch_2019]) and multiple hardware (e.g., CPU, GPU).
Our packages fully support vectorization to leverage the performance of these array libraries.

# Acknowledgements

This work used computational resources
Supermicro ARS-111GL-DNHR-LCC and FUJITSU Server PRIMERGY CX2550 M7 (Miyabi) at Joint Center for Advanced High Performance Computing (JCAHPC) and
TSUBAME4.0 supercomputer provided by Institute of Science Tokyo
through Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures and High Performance Computing Infrastructure in Japan (Project ID: jh240031).

# References
