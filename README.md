# `qittoolbox`, a small Quantum Information Theory toolbox

## Introduction
This small Quantum Information Theory toolbox includes tools to analyze information-theoretic quantities of quantum channels. Broadly speaking, it includes:
1. An implementation of so-called $O_N^+$-Clebsch Gordan quantum channels introduced in `[BC16]`, and the $S_N^+$-Clebsch Gordan quantum channels whose description can be found in my thesis `[SvP22]`, based on Kraus operators and Choi matrices, represented using `scipy.sparse` matrices where applicable, and `numpy.ndarray`'s otherwise. This allows the user to get matrix representations of the so-called _Jones-Wenzl projections_ , matrix representations of the Kraus operators of these quantum channels, and the Choi matrices associated to these quantum channels.

2. An implementation of the _DPS hierarchy_ `[DPS04]` and _DPS* criterion_ `[NOP09]` , which are methods to check whether a quantum mixed state $\rho \in \mathcal{S}(\mathcal{H}_\text{A} \otimes \mathcal{H}_\text{B})$ (with finite-dimensional Hilbert spaces $\mathcal{H}_\text{A}$ and $\mathcal{H}_\text{B}$) is _separable_. When applied to the Choi matrix of a quantum channel, separability tells you whether the quantum channel is _entanglement-breaking_. 

3. Some implementations of the derivative-free optimization technique called _Particle Swarm Optimization_ , which can be utilized to seek good approximations of the global minimum of some information-theoretic quantity you are interested in, such as the _minimum output entropy_ .
    1. A traditional implementation on a search domain $[a_1,b_1]\times \cdots \times [a_n,b_n] \subseteq \mathbb{R}^n$ with $-\infty \leq a_i < b_i \leq \infty$.
    2. An implementation of this optimization where the search domain is the unit sphere $\mathbb{S}^{n-1} \subset \mathbb{R}^{n}$ for some $n \in \mathbb{N}$. 
    3. An implementation of this optimization where _gradient-descent_ is used to optimize the global best position after each timestep.

4. An implementation of an $\epsilon$-cover over the unit sphere $\mathbb{S}^{n-1} \subset \mathbb{R}^n$, which is a subset $A \subseteq \mathbb{S}^{n-1}$ such that for any unit vector $\ket{\psi} \in \mathbb{S}^{n-1}$, one can find a unit vector $\ket{\phi} \in A$ such that $\Vert \psi - \phi \Vert_2 \leq \epsilon$. This, in combination with a continuity bound such as Audenaert's continuity bound for the von Neumann entropy, can give a lower bound for a minimization problem involving an information-theoretic quantity such as the _minimum output entropy_ . 

4. Some elementary finite-dimensional linear algebra manipulations implemented as small wrappers around `scipy` and `numpy` calls, such as tensor leg permutations, an easy way to get (sparse) vectors representing quantum states such as $\ket{0321}$.


# How to use `qittoolbox`

## Prerequisites
 - A Python installation with `version >= 3.6`.
 - The following packages, that can be installed by running the command `python -m pip install <package_name>` in a console, where `<package_name>` is the following: 
    -    `numpy`
    -   `scipy` 
    - Either `cvxpy` or `picos`, or both.

## Installing `qittoolbox` using a Python wheel
The pre-built Python wheel `qittoolbox-0.0.1-py3-none-any.whl` is available in the sub-directory `/dist/`. After copying this file to your local machine, open a command window and run the command

    python -m pip install qittoolbox-0.0.1-py3-none-any.whl

You can uninstall the package by running `python -m pip uninstall qittoolbox` .  

## Installing `qittoolbox` by building the package
Copy the entire GitHub repository to your local machine. Make sure that your Python installation has `setuptools` installed. Then, open a console and run the command

    python setup.py bdist_wheel

to build the project, which should yield a subfolder `/dist/` where you will find a wheel (`.whl` file), which you can install by following the instructions from the previous section.

## Using `qittoolbox` without installing it as a package
Copy the entire GitHub repository to your local machine. If your local folder structure looks like `../parent_dir/qittoolbox/qittoolbox/(...)`, then you can import `qittoolbox` if your Python program is running in the `../parent_dir/qittoolbox` directory.

## How to use: the $O_N^+$- and $S_N^+$-quantum channels
See [the tutorial on $O_N^+$ and $S_N^+$ channels](tutorials/tutorial_ONplus_and_SNplus_channels.ipynb)

## How to use: the DPS hierarchy and DPS* criterion
See [the tutorial on the DPS hierarchy and the DPS\* criterion](tutorials/tutorial_DPS_and_DPSstar.ipynb)

## How to use: the Particle Swarm Optimization technique
See [the tutorial on the PSO algorithm](tutorials/tutorial_Particle_Swarm_Optimization.ipynb)

## How to use: the $\epsilon$-covers
See [the tutorial on the $\epsilon$-covers](tutorials/tutorial_epsilon_covers.ipynb)

## Bibliography
- `[BC16]` M. Brannan and B. Collins. _Highly Entangled, Non-Random Subspaces of Tensor Products from Quantum Groups._ Comm. Math. Phys. 358 (2018), no. 3, 1007-1025, 358(3):1007–1025, December 2016, _arXiv_ : 1612.09598. doi:10.1007/s00220-017-3023-6.
- `[DPS04]` A. C. Doherty, P. A. Parrilo, and F. M. Spedalieri. _A Complete Family of Separability Criteria._ Physical Review A, 69(2):022308, feb 2004. doi:10.1103/physreva.69.022308.
- `[NOP09]` M. Navascués, M. Owari, and M. B. Plenio. _Complete Criterion for Separability Detection._ Physical Review Letters, 103(16):160404, oct 2009. doi:10.1103/physrevlett.103.160404.
- `[SvP22]` S. van Poelgeest. _Information-theoretic Quantities of Quantum Channels with Partition Quantum Group Symmetries_ . Master thesis. 