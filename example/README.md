# OpTaS Examples

Several examples are contained in this directory.

The majority of these examples require [PyBullet](https://pybullet.org/wordpress/), and some use the OpTaS visualizer, which requires [VTK](https://vtk.org/).
Install everything the examples need with `$ pip install .[example]` from the repository root.

## Examples

### Position IK

The `example.py` script showcases some of the main features of OpTaS.

### Figure of Eight Planning

This script, `figure_eight_plan.py`, demonstrates how to implement a simple planner.
A hand-crafted task space trajectory is given, and the planner finds the joint space trajectory.

### Dual Arm

The script that runs the dual arm example is `dual_arm.py`.
The goal of this example is to demonstrate how multiple robots can be incorporated into a single optimization formulation.

A task space trajectory is hand crafted, for a dual arm setup using two KUKA robot arms.
A joint space plan is computed for both arms by solving an optimization problem in a single formulation.

### Visualization

This is a simple example that shows how to pass the robot model to the visualizer.
The script that implements this example is `visualization.py`.

### MPC

A basic task space MPC controller is demonstrated in the `point_mass_mpc.py` script.
The script implements a basic MPC controller, and plots the solution in a short animation.

The demonstration first computes a task space plan.
Then the robot (point mass) is tasked with tracking the plan whilst adapting to dynamic obstacles in the scene.
An obstacle, at execution time, is given some motion that impedes the plan - the goal of the MPC is to avoid the obstacle.
This is achieved via the MPC controller.

### Point Mass Planning

This example, in the script `point_mass_planner.py`, demonstrates how to plan a trajectory using a task model.

### Planar IK

The `planar_ik.py` script contains a simple example of a possible full Inverse Kinematics (IK) implementation for a planar three Degrees of Freedom (DoF) robot arm.
The task-space consists of the x and y position of the end-effector and the IK computes an optimal set of joint angles, using a non-linear program formulation solved with `CasADiSolver` (IPOPT).

### Planar IDK

The `planar_idk.py` script contains a simple example of a possible Inverse Differential Kinematics (IDK) implementation for a planar three DoF robot arm.
The task-space consists of a desired displacement of the x and y position of the end-effector and the IDK computes the optimal set of joint angle displacements, using two different alternative Quadratic Program (QP) solvers.
This formulation also adds bounds to the end-effector orientation and compares the result with the solution obtained through the pseudo-inverse approach.

