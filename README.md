<p align="center">
  <img src="https://raw.githubusercontent.com/joaomoura24/optas/master/doc/logo.png" width="60" align="right">
</p>

# OpTaS

> **Note:** This repository is based on the original [cmower/optas](https://github.com/cmower/optas), but has significantly diverged from it.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/joaomoura24/optas/actions/workflows/ci.yaml/badge.svg)](https://github.com/joaomoura24/optas/actions/workflows/ci.yaml)
[![Docs](https://github.com/joaomoura24/optas/actions/workflows/documentation.yaml/badge.svg)](https://github.com/joaomoura24/optas/actions/workflows/documentation.yaml)


OpTaS (OPtimization-based TAsk Specification) is a light layer over [CasADi](https://web.casadi.org/) for robot optimization: it loads robots from URDF, gives their kinematics as CasADi expressions, and turns costs and constraints on them into CasADi problems for inverse kinematics, trajectory optimization and model predictive control.

- Code: [https://github.com/joaomoura24/optas](https://github.com/joaomoura24/optas)
- Documentation: [https://joaomoura24.github.io/optas/](https://joaomoura24.github.io/optas/)
- Issues: [https://github.com/joaomoura24/optas/issues](https://github.com/joaomoura24/optas/issues)
- ICRA 2023 paper:
  - (arXiv) [https://arxiv.org/abs/2301.13512](https://arxiv.org/abs/2301.13512)
  - (ieee) [https://ieeexplore.ieee.org/document/10161272](https://ieeexplore.ieee.org/document/10161272)
- Video: [https://youtu.be/gCMNOenFngU](https://youtu.be/gCMNOenFngU)
- Presentation: [https://vimeo.com/824802366](https://vimeo.com/824802366)

In the past, OpTaS supported ROS from an internal module. This functionality, with additional updates, has now been moved to a dedicated repository: [optas_ros](https://github.com/cmower/optas_ros).

# Example

<p align="center">
  <img src="https://raw.githubusercontent.com/joaomoura24/optas/master/doc/image/kuka_example.png" width="61.803398875%">
</p>


In this example we implement an optimization-based IK problem.
The problem computes an optimal joint configuration $q^*\in\mathbb{R}^n$ given by

$$
q^* = \underset{q}{\text{arg}\min}~\|\|q - q_N\|\|^2\quad\text{subject to}\quad p(q) = p_g, q^-\leq q \leq q^+
$$

where
* $q\in\mathbb{R}^n$ is the joint configuration for an $n$-dof robot (in our example, we use the KUKA LWR in the above figure with $n=7$),
* $q_N\in\mathbb{R}^n$ is a nominal joint configuration,
* $\|\|\cdot\|\|$ is the Euclidean norm,
* $p: \mathbb{R}^n\rightarrow\mathbb{R}^3$ computes the end-effector position via the forward kinematics,
* $p_g\in\mathbb{R}^3$ is a goal position, and
* $q^-, q^+\in\mathbb{R}^n$ is the lower and upper joint position limits respectively.

The example problem has a quadratic cost function with nonlinear constraints.
We use the nominal configuration $q_N$ as the initial seed for the problem.

The following example script showcases some of the main features of OpTaS:
creating a robot model,
building an optimization problem,
passing the problem to a solver,
computing an optimal solution, and
visualizing the robot in a given configuration.

```python
import os
import pathlib

import optas

# Specify URDF filename
cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
urdf_filename = os.path.join(
    cwd, "robots", "kuka_lwr", "kuka_lwr.urdf"
)  # KUKA LWR, 7-DoF

# Setup robot model
robot = optas.RobotModel(urdf_filename=urdf_filename)
name = robot.get_name()

# Setup optimization builder
T = 1
builder = optas.OptimizationBuilder(T, robots=robot)

# Setup parameters
qn = builder.add_parameter("q_nominal", robot.ndof)
pg = builder.add_parameter("p_goal", 3)

# Constraint: end goal
q = builder.get_model_state(name, 0)
end_effector_name = "end_effector_ball"
p = robot.get_global_link_position(end_effector_name, q)
builder.add_equality_constraint("end_goal", p, pg)

# Cost: nominal configuration
builder.add_cost_term("nominal", optas.sumsqr(q - qn))

# Constraint: joint position limits
builder.enforce_model_limits(name)  # joint limits extracted from URDF

# Build optimization problem
optimization = builder.build()

# Interface optimization problem with a solver
solver = optas.CasADiSolver(optimization).setup("ipopt")
# solver = optas.ScipyMinimizeSolver(optimization).setup("SLSQP")

# Specify a nominal configuration
q_nominal = optas.deg2rad([0, 45, 0, -90, 0, -45, 0])

# Get end-effector position in nominal configuration
p_nominal = robot.get_global_link_position(end_effector_name, q_nominal)

# Specify a goal end-effector position
p_goal = p_nominal + optas.DM([0.0, 0.3, -0.2])

# Reset solver parameters
solver.reset_parameters({"q_nominal": q_nominal, "p_goal": p_goal})

# Reset initial seed
solver.reset_initial_seed({f"{name}/q": q_nominal})

# Compute a solution
solution = solver.solve()
q_solution = solution[f"{name}/q"]

# Visualize the robot
vis = optas.Visualizer(quit_after_delay=2.0)

# Draw goal position and start visualizer
vis.sphere(0.05, rgb=[0, 1, 0], position=p_goal.toarray().flatten().tolist())
# vis.robot(robot, q=q_nominal,display_link_names=True,show_links=True)   # nominal
vis.robot(robot, q=q_solution, display_link_names=True, show_links=True)  # solution

vis.start()
```

Run the example script [example.py](example/example.py).
Other examples, including dual-arm planning, Model Predictive Control, Trajectory Optimization, etc can be found in the [example/](example) directory.

## Code generation

A problem solved with `CasADiSolver` can be turned into C and compiled (needs `gcc`), e.g. to call it from a C++ controller:

```python
# The example has nonlinear constraints: SQP with qrqp for the QP steps (for a QP: daqp, qrqp or highs)
solver = optas.CasADiSolver(optimization).setup("sqpmethod", {"qpsol": "qrqp"})
solver.reset_parameters({"q_nominal": q_nominal})  # parameters that are not inputs keep these values
solver.reset_initial_seed({f"{name}/q": q_nominal})
F = solver.to_function("ik", inputs=["p_goal"], outputs=[f"{name}/q/x"])
library = solver.generate(F, "build")  # build/ik.c, build/ik.h, build/libik.so
```

`to_function` returns a CasADi function that can also be called from Python. Load the library with `casadi.external("ik", library)`, or with `dlopen` through the CasADi C API in `ik.h`. CasADi can generate code for some solvers only (e.g. qrqp, daqp, highs; not qpoases, proxqp or ipopt).

# Support

Tested on Ubuntu 24.04 with Python 3.12 (see the [CI workflow](.github/workflows/ci.yaml)).

# Install

Make sure `pip` is up-to-date by running `$ python -m pip install --upgrade pip`.

## Via pip

```
$ python -m pip install 'pyoptas @ git+https://github.com/joaomoura24/optas.git'
```

Note that `pip install pyoptas` installs the original upstream package from PyPI, not this version.

The core install only needs CasADi, NumPy, SciPy, urdf-parser-py and PyYAML.
Optional features are available as extras, e.g. `'pyoptas[vis,solvers] @ git+...'`:

| Extra | Adds | Needed for |
|---|---|---|
| `vis` | vtk | `optas.Visualizer` |
| `solvers` | osqp, cvxopt | `OSQPSolver`, `CVXOPTSolver` |
| `xacro` | xacro | loading models from xacro files |
| `all` | all of the above | |

## From source
1. `$ git clone --recursive https://github.com/joaomoura24/optas.git` (if you do not want to build the documentation then the `--recursive` flag is not necessary)
2. `$ cd optas`
4. `$ pip install .` (or `$ pip install .[all]` for all optional features)
   - if you want to run the examples use: `$ pip install .[example]`
   - if you want to run the tests use: `$ pip install .[test]`

### Build documentation

1. `$ cd /path/to/optas/doc`
2. `$ sudo apt install doxygen graphviz`
3. `$ python gen_mainpage.py`
3. `$ doxygen`
4. Open the documentation in either HTML or PDF:
   - `html/index.html`
   - `latex/refman.pdf`

### Run tests

1. `$ cd /path/to/optas`
2. Each test can be run as follows
   - `$ pytest tests/test_builder.py`
   - `$ pytest tests/test_examples.py`
   - `$ pytest tests/test_models.py`
   - `$ pytest tests/test_optas_utils.py`
   - `$ pytest tests/test_optimization.py`
   - `$ pytest tests/test_solver.py`
   - `$ pytest tests/test_spatialmath.py`
   - `$ pytest tests/test_sx_container.py`

Example tests that simulate in PyBullet are skipped unless it is installed (`$ pip install pybullet`).

# Known Issues

- Loading robot models from xacro files is supported, however there can be issues if you are running this in a ROS agnositic environment. If you do not have ROS installed, then the xacro file should not contain ROS-specific features. For further details see [here](https://github.com/cmower/optas/issues/78).
- If NumPy ver 1.24 is installed, an `AttributeError` error is thrown when you try to solve an unconstrained problem with the OSQP interface. A temporary workaround is to add a constraint, e.g. `x >= -1e9` where `x` is a decision variable. See details on the issue [here](https://github.com/osqp/osqp-python/issues/104) and pull request [here](https://github.com/osqp/osqp-python/pull/105).

# Citation

If you use OpTaS in your work, please consider including the following citation.

```bibtex
@inproceedings{mower23,
  author={Mower, Christopher E. and Moura, Jo\~{a}o and Behabadi, Nazanin Zamani and Vijayakumar, Sethu and Vercauteren, Tom and Bergeles, Christos},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  title={OpTaS: An Optimization-based Task Specification Library for Trajectory Optimization and Model Predictive Control},
  year={2023},
  doi={10.1109/ICRA48891.2023.10161272}
}
```

The preprint can be found on [arXiv](https://arxiv.org/abs/2301.13512).

# Contributing

We welcome contributions from the community.
If you come across any issues or inacuracies in the documentation, please [submit an issue](https://github.com/joaomoura24/optas/issues/new/choose).
If you would like to contribute any new features, please [fork the repository](https://github.com/joaomoura24/optas/fork), and submit a pull request.

# Acknowledgement

This research received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No. 101016985 ([FAROS](https://h2020faros.eu/)).
Further, this work was supported by core funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1].
T. Vercauteren is supported by a Medtronic / RAEng Research Chair [RCSRF1819\7\34], and C. Bergeles by an ERC Starting Grant [714562].
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101017008, Enhancing Healthcare with Assistive Robotic Mobile Manipulation ([HARMONY](https://harmony-eu.org/)).

<p align="center">
  <img src="https://raw.githubusercontent.com/joaomoura24/optas/master/doc/image/eu.png" width="180" align="left">
</p>
