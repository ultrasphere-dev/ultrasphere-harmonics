# Detailed Explanation

Let

$$
P_n (\mathbb{R}^d) := \left\{\sum_{|\alpha| = n - 1} c_\alpha x^\alpha | c_\alpha \in \mathbb{C}\right\}
$$

space of homogeneous polynomials,

$$
H_n (\mathbb{R}^d) := \left\{f \in P_n (\mathbb{R}^d) | \Delta f = 0\right\}
$$

space of homogeneous harmonic polynomials, and

$$
H_n (\mathbb{S}^{d-1}) := \left\{f|_{\mathbb{S}^{d-1}} | f \in H_n (\mathbb{R}^d)\right\}
$$

space of spherical harmonics.

We are interested to compute orthonormal basis of $H_n (\mathbb{S}^{d-1})$ which coressponds to one of spherical coordinates.

It is possible to construct such basis $\{Y^{n,p}\}_{p=1}^{\dim H_n (\mathbb{S}^{d-1})}$ so that any element $Y^{n,p}$ in the basis can be expressed as product of eigenfunctions $\psi^{n,p}_\Theta$ for each node $\Theta$ in the tree representation of spherical coordinates:

$$
Y^l(\theta) = \prod_{\Theta \in \text{Nodes}} \psi^{n,p}_\Theta (\theta_{\Theta})
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
