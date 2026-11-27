import os
import pathlib

import numpy as np
import pytest

import optas

pytest.importorskip("viser")

from optas.visualize import Visualizer, _alphas, _wxyz  # noqa: E402

ROBOT = os.path.join(
    pathlib.Path(__file__).parent.parent.resolve(),
    "example",
    "robots",
    "kuka_lwr",
    "kuka_lwr.urdf",
)


@pytest.fixture
def vis():
    v = Visualizer(port=8090)
    yield v
    v.server.stop()


@pytest.fixture(scope="module")
def robot():
    return optas.RobotModel(urdf_filename=ROBOT)


def link_index(robot, name):
    return [link.name for link in robot.get_urdf().links].index(name)


def handles_of_link(handles, index):
    return [h for h in handles if h._optas_link[0] == index]


def test_wxyz():
    # quaternions are given as (x, y, z, w), as in optas/scipy; viser takes (w, x, y, z)
    assert np.allclose(_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])
    assert np.allclose(
        _wxyz([0, 0, 90], euler_degrees=True), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    )
    with pytest.raises(ValueError):
        _wxyz([1, 2])


def test_alphas():
    assert _alphas(None, 3) == [1.0, 1.0, 1.0]
    assert np.allclose(_alphas({"style": "A"}, 3), [0.1, 0.55, 1.0])
    assert _alphas({"style": "B"}, 3) == [0.1, 0.1, 1.0]
    assert _alphas({"style": "C", "alpha_mid": 0.2}, 4) == [1.0, 0.2, 0.2, 1.0]
    assert _alphas({"style": "const", "alpha": 0.3}, 2) == [0.3, 0.3]
    with pytest.raises(ValueError):
        _alphas({"style": "?"}, 2)


def test_robot_poses(vis, robot):
    q = robot.get_random_joint_positions()
    handles = vis.robot(robot, q, show_links=True, display_link_names=True)
    nlinks = len(robot.get_urdf().links)
    nvisuals = len(vis._visuals(robot).visuals)
    assert len(handles) == nvisuals + 2 * nlinks

    # the link frame of the end effector is at its forward kinematics
    T = robot.get_global_link_transform("end_effector_ball", q).toarray()
    index = link_index(robot, "end_effector_ball")
    positions = [h.position for h in handles_of_link(handles, index)]
    assert any(np.allclose(p, T[:3, 3]) for p in positions)

    # a visual origin is applied once (lwr_arm_1_link visual is at z = -0.008 in the link)
    index = link_index(robot, "lwr_arm_1_link")
    T = robot.get_global_link_transform("lwr_arm_1_link", q).toarray()
    expected = T[:3, 3] + T[:3, :3] @ [0, 0, -0.008]
    (mesh,) = [h for h in handles_of_link(handles, index) if hasattr(h, "opacity")]
    assert np.allclose(mesh.position, expected)


def test_robot_traj_animation(vis, robot):
    n = 11
    Q = np.linspace(np.zeros(robot.ndof), 0.5 * np.ones(robot.ndof), n).T
    handles = vis.robot_traj(robot, Q, animate=True, duration=2.0)
    index = link_index(robot, "end_effector_ball")
    (ee,) = handles_of_link(handles, index)

    for t, q in [(0.0, Q[:, 0]), (1.0, Q[:, 5]), (2.0, Q[:, -1]), (9.0, Q[:, -1])]:
        vis.show_time(t)
        T = robot.get_global_link_transform("end_effector_ball", q).toarray()
        assert np.allclose(ee.position, T[:3, 3])

    # without animation: one robot per configuration
    nvisuals = len(vis._visuals(robot).visuals)
    ghosts = vis.robot_traj(robot, Q[:, :3], alpha_spec={"style": "A"})
    assert len(ghosts) == 3 * nvisuals
    assert ghosts[0].opacity == pytest.approx(0.1) and ghosts[-1].opacity is None


def test_sphere_traj(vis):
    traj = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    (sphere,) = vis.sphere_traj(traj, radius=0.1, animate=True, duration=1.0)
    vis.show_time(0.5)
    assert np.allclose(sphere.position, [1.0, 0.0, 0.0])
    assert len(vis.sphere_traj(traj, radius=0.1)) == 3


def test_shapes(vis):
    box = vis.box(scale=[1, 2, 3], rgb=[1, 0, 0], alpha=0.5, position=[1, 2, 3])
    assert np.allclose(box.position, [1, 2, 3]) and box.opacity == 0.5
    assert vis.cylinder(radius=0.1, height=0.5) is not None
    assert vis.line([0, 0, 0], [1, 1, 1]) is not None
    frame = vis.link(optas.DM.eye(4))
    assert np.allclose(frame.wxyz, [1, 0, 0, 0])
    assert vis.text("hello", [0, 0, 1]) is not None
    assert vis.grid_floor(num_cells=4, stride=0.5) is not None
    example = pathlib.Path(__file__).parent.parent / "example"
    duck = vis.mesh(
        str(example / "duck.obj"), texture_filename=str(example / "duckCM.png")
    )
    assert duck is not None
    with pytest.warns(UserWarning, match="skipping mesh"):
        assert vis.mesh("missing.stl") is None


def test_start_returns_after_delay(robot):
    vis = Visualizer(port=8090, quit_after_delay=0.3)
    vis.robot_traj(robot, np.zeros((robot.ndof, 2)), animate=True, duration=1.0)
    vis.start()  # plays the animation, then stops the server
