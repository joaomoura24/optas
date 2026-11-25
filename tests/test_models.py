import os
import pytest
import optas
import pathlib
import numpy as np
import scipy.linalg as linalg
import urdf_parser_py.urdf as urdf
from scipy.spatial.transform import Rotation as Rot
from .tester_robot_model import RobotModelTester

NUM_RANDOM = 100

TESTS_PATH = pathlib.Path(__file__).parent.resolve()
EXAMPLE_PATH = TESTS_PATH.parent / "example"

###########################################################################
# Helper methods
#


def random_vector(lo=-1, hi=1, n=3):
    return np.random.uniform(-lo, hi, size=(n,))


def isclose(A: np.ndarray, B: np.ndarray, atol=1e-08):
    return np.isclose(A, B, atol=atol).all()


def wrap(a):
    """Wrap angles to [-pi, pi): pi and -pi are the same angle."""
    return (a + np.pi) % (2 * np.pi) - np.pi


###########################################################################
# Tests
#


def test_listify_output():
    class _test_cls:
        @optas.models.listify_output
        def method(self, link, q):
            return 2.0 * q

    tc = _test_cls()

    q = optas.DM([1, 2, 3])
    q_exp = optas.DM([2, 4, 6])
    assert isclose(q_exp.toarray(), tc.method("test", q).toarray())

    q = optas.DM(
        [
            [1, 5],
            [2, 4],
            [3, 3],
        ]
    )
    q_exp = 2.0 * q

    assert isclose(q_exp.toarray(), tc.method("test", q).toarray())


class Test_Model:
    def test_get_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.get_name() == name

    def test_state_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.state_name(0) == "test/x"
        assert model.state_name(1) == "test/dx"

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.state_name(n)

    def test_state_parameter_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.state_parameter_name(0) == "test/x/p"
        assert model.state_parameter_name(1) == "test/dx/p"

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.state_parameter_name(n)

    def test_get_limits(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        for td, (lo_exp, hi_exp) in dlim.items():
            lo, hi = model.get_limits(td)
            assert isclose(lo, lo_exp)
            assert isclose(hi, hi_exp)

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.get_limits(n)

    def test_in_limit(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        def in_limit_known(x, td):
            lo, hi = dlim[td]
            return np.all(np.logical_and(lo <= x, x <= hi))

        for td in dlim.keys():
            for _ in range(NUM_RANDOM):
                x = np.random.uniform(-1.5, 1.5, size=(dim,))
                print(in_limit_known(x, td))
                print(model.in_limit(x, td).toarray().flatten())
                assert (
                    in_limit_known(x, td) == model.in_limit(x, td).toarray().flatten()
                )


def test_TaskModel():
    name = "test"
    dim = 3
    dlim = {
        0: ([-1, -1, -1], [1, 1, 1]),
    }
    T = 2

    model = optas.TaskModel(name, dim, dlim=dlim, T=T)
    assert model.state_name(0) == "test/y"
    assert model.state_parameter_name(0) == "test/y/p"
    assert model.T == T


def test_TaskModel_discrete():
    name = "test"
    dim = 3
    dlim = {
        0: ([-1, -1, -1], [1, 1, 1]),
    }
    T = 2
    is_discrete = True

    model = optas.TaskModel(name, dim, dlim=dlim, T=T, is_discrete=is_discrete)
    assert model.state_name(0) == "test/y"
    assert model.state_parameter_name(0) == "test/y/p"
    assert model.T == T
    assert model.is_discrete == is_discrete


class TestRobotModel:
    # Setup path to tester robot URDF
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_filename = "tester_robot.urdf"
    urdf_path = os.path.join(cwd, urdf_filename)

    # Setup model (with no parameterized joints)
    model = optas.RobotModel(urdf_filename=urdf_path)

    model_expected_joint_names = [
        "joint0",
        "joint1",
        "joint2",
        "eff_joint",
    ]
    model_expected_link_names = ["world", "link1", "link2", "link3", "eff"]
    model_expected_actuated_joint_names = ["joint0", "joint1", "joint2"]
    model_expected_parameter_joint_names = []
    model_expected_optimized_joint_indexes = [0, 1, 2]
    model_expected_optimized_joint_names = ["joint0", "joint1", "joint2"]
    model_expected_parameter_joint_indexes = []

    model_expected_lower_actuated_joint_limits = [-1e9, -1, 0]
    model_expected_upper_actuated_joint_limits = [1e9, 1, 1]
    model_expected_velocity_actuated_joint_limits = [1e9, 1, 1]
    model_expected_lower_optimized_joint_limits = [-1e9, -1, 0]
    model_expected_upper_optimized_joint_limits = [1e9, 1, 1]
    model_expected_velocity_optimized_joint_limits = [1e9, 1, 1]

    # Setup model (with parameterized joints)
    model_p = optas.RobotModel(
        urdf_filename=os.path.join(cwd, urdf_filename),
        param_joints=["joint0"],
    )

    model_p_expected_joint_names = [
        "joint0",
        "joint1",
        "joint2",
        "eff_joint",
    ]
    model_p_expected_link_names = ["world", "link1", "link2", "link3", "eff"]
    model_p_expected_actuated_joint_names = ["joint0", "joint1", "joint2"]
    model_p_expected_parameter_joint_names = ["joint0"]
    model_p_expected_optimized_joint_indexes = [1, 2]
    model_p_expected_optimized_joint_names = ["joint1", "joint2"]
    model_p_expected_parameter_joint_indexes = [0]

    model_p_expected_lower_actuated_joint_limits = [-1e9, -1, 0]
    model_p_expected_upper_actuated_joint_limits = [1e9, 1, 1]
    model_p_expected_velocity_actuated_joint_limits = [1e9, 1, 1]
    model_p_expected_lower_optimized_joint_limits = [-1, 0]
    model_p_expected_upper_optimized_joint_limits = [1, 1]
    model_p_expected_velocity_optimized_joint_limits = [1, 1]

    def test_init(self):
        with pytest.raises(AssertionError):
            model = optas.RobotModel()

    def test_get_urdf(self):
        assert isinstance(self.model.get_urdf(), urdf.Robot)

    def test_get_urdf_dirname(self):
        urdf_dirname = self.model.get_urdf_dirname()
        assert isinstance(urdf_dirname, pathlib.Path)
        assert urdf_dirname == self.cwd

    def check_items(self, items_label):
        for model_name in ("model", "model_p"):
            model = getattr(self, model_name)
            items = getattr(model, items_label)

            expected_items_label = model_name + "_expected_" + items_label
            expected_items = getattr(self, expected_items_label)

            if isinstance(items, list):
                assert len(items) == len(expected_items)
                for item, expected_item in zip(items, expected_items):
                    assert item == expected_item
            elif isinstance(items, optas.DM):
                assert items.shape[0] == len(expected_items)
                for i in range(len(expected_items)):
                    assert isclose(items[i].toarray().flatten(), expected_items[i])

    def test_joint_names(self):
        self.check_items("joint_names")

    def test_link_names(self):
        self.check_items("link_names")

    def test_actuated_joint_names(self):
        self.check_items("actuated_joint_names")

    def test_parameter_joint_names(self):
        self.check_items("parameter_joint_names")

    def test_optimized_joint_indexes(self):
        self.check_items("optimized_joint_indexes")

    def test_optimized_joint_names(self):
        self.check_items("optimized_joint_names")

    def test_parameter_joint_indexes(self):
        self.check_items("parameter_joint_indexes")

    def test_extract_parameter_dimensions(self):
        q = optas.DM([1, 2, 3])
        qp = self.model_p.extract_parameter_dimensions(q)
        assert isclose(q[0].toarray().flatten(), qp.toarray().flatten())

    def test_extract_optimized_dimensions(self):
        q = optas.DM([1, 2, 3])
        qopt = self.model_p.extract_optimized_dimensions(q)
        assert isclose(q[1:].toarray().flatten(), qopt.toarray().flatten())

    def check_num(self, label, expected_nums):
        for model_name, expected_num in zip(("model", "model_p"), expected_nums):
            model = getattr(self, model_name)
            assert getattr(model, label) == expected_num

    def test_ndof(self):
        self.check_num("ndof", [3, 3])

    def test_num_opt_joints(self):
        self.check_num("num_opt_joints", [3, 2])

    def test_num_param_joints(self):
        self.check_num("num_param_joints", [0, 1])

    def test_get_joint_lower_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_joint_lower_limit(joint_null), -1e9)

        expected_limits = (-1e9, -1, 0, -1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_joint_lower_limit(joint)
            assert isclose(limit, expected_limit)

    def test_get_joint_lower_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_joint_upper_limit(joint_null), 1e9)

        expected_limits = (1e9, 1, 1, 1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_joint_upper_limit(joint)
            assert isclose(limit, expected_limit)

    def test_get_velocity_joint_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_velocity_joint_limit(joint_null), 1e9)

        expected_limits = (1e9, 1, 1, 1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_velocity_joint_limit(joint)
            assert isclose(limit, expected_limit)

    def test_lower_actuated_joint_limits(self):
        self.check_items("lower_actuated_joint_limits")

    def test_upper_actuated_joint_limits(self):
        self.check_items("upper_actuated_joint_limits")

    def test_velocity_actuated_joint_limits(self):
        self.check_items("velocity_actuated_joint_limits")

    def test_lower_optimized_joint_limits(self):
        self.check_items("lower_optimized_joint_limits")

    def test_upper_optimized_joint_limits(self):
        self.check_items("upper_optimized_joint_limits")

    def test_velocity_optimized_joint_limits(self):
        self.check_items("velocity_optimized_joint_limits")

    def test_add_base_frame(self):
        model = optas.RobotModel(urdf_filename=self.urdf_path)
        assert model.get_root_link() == "world"

        model.add_base_frame(
            "test_world",
            xyz=[1, 2, 3],
            rpy=[0, 0, 0.5 * np.pi],
            joint_name="test_joint_name",
        )
        assert model.get_root_link() == "test_world"

    def test_get_root_link(self):
        assert self.model.get_root_link() == "world"
        assert self.model_p.get_root_link() == "world"

    def test_get_link_visual_origin(self):
        for link in self.model.get_urdf().links:
            xyz, rpy = self.model.get_link_visual_origin(link)
            assert isclose(xyz.toarray().flatten(), np.zeros(3))
            assert isclose(rpy.toarray().flatten(), np.zeros(3))

        for link in self.model_p.get_urdf().links:
            xyz, rpy = self.model_p.get_link_visual_origin(link)
            assert isclose(xyz.toarray().flatten(), np.zeros(3))
            assert isclose(rpy.toarray().flatten(), np.zeros(3))

    def test_get_joint_origin(self):
        expected_joint_origins = [
            ([0, 0, 0], [0, 0, 0]),
            ([2, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [0, 0, 0]),
            ([0, 0, 0.5], [0, 0, 0]),
        ]

        for joint, (expected_xyz, expected_rpy) in zip(
            self.model.get_urdf().joints, expected_joint_origins
        ):
            xyz, rpy = self.model.get_joint_origin(joint)
            assert isclose(xyz.toarray().flatten(), expected_xyz)
            assert isclose(rpy.toarray().flatten(), expected_rpy)

        for joint, (expected_xyz, expected_rpy) in zip(
            self.model_p.get_urdf().joints, expected_joint_origins
        ):
            xyz, rpy = self.model_p.get_joint_origin(joint)
            assert isclose(xyz.toarray().flatten(), expected_xyz)
            assert isclose(rpy.toarray().flatten(), expected_rpy)

    def test_get_joint_axis(self):
        expected_joint_axis = [0, 0, 1]

        joints = self.model.get_urdf().joints
        for joint in joints[:-1]:
            joint_axis = self.model.get_joint_axis(joint)
            assert isclose(joint_axis.toarray().flatten(), expected_joint_axis)
        joint_axis = self.model.get_joint_axis(joints[-1])
        assert isclose(joint_axis.toarray().flatten(), [1, 0, 0])

        joints = self.model_p.get_urdf().joints
        for joint in joints[:-1]:
            joint_axis = self.model_p.get_joint_axis(joint)
            assert isclose(joint_axis.toarray().flatten(), expected_joint_axis)
        joint_axis = self.model_p.get_joint_axis(joints[-1])
        assert isclose(joint_axis.toarray().flatten(), [1, 0, 0])

    def test_get_actuated_joint_index(self):
        expected_joint_indexes = [0, 1, 2]

        for joint_name, expected_joint_index in zip(
            self.model.actuated_joint_names, expected_joint_indexes
        ):
            joint_index = self.model.get_actuated_joint_index(joint_name)
            assert joint_index == expected_joint_index

        for joint_name, expected_joint_index in zip(
            self.model_p.actuated_joint_names, expected_joint_indexes
        ):
            joint_index = self.model_p.get_actuated_joint_index(joint_name)
            assert joint_index == expected_joint_index

    def test_get_random_joint_positions(self):
        for _ in range(NUM_RANDOM):
            qr = self.model.get_random_joint_positions()
            assert isclose(self.model.in_limit(qr, 0).toarray().flatten(), 1)

    def test_get_random_pose_in_global_link(self):
        with pytest.raises(TypeError):
            self.model.get_random_pose_in_global_link()

        max_radius = np.linalg.norm([3, 0, 1.5])  # prismatic joint at upper limit

        for _ in range(NUM_RANDOM):
            for link_name in self.model.link_names:
                T = self.model.get_random_pose_in_global_link(link_name)
                assert np.linalg.norm(T[:3, 3].toarray().flatten()) <= max_radius

    def test_get_global_link_transform_function_numpy_output(self):
        link_name = self.model.link_names[-1]
        q = self.model.get_random_joint_positions().toarray()
        Tr = self.model.get_global_link_transform_function(link_name, numpy_output=True)
        T = Tr(q)
        assert isinstance(T, np.ndarray)

        q = optas.SX.sym("q", q.shape[0])
        with pytest.raises(AssertionError):
            T = Tr(q)


class Robot:
    """Model under test, its roboticstoolbox reference, and a link/base pair for
    relative checks. The base must not be adjacent to the link: a constant offset
    can't tell inv(T_B) @ T_L apart from other compositions of the two transforms."""

    def __init__(self, urdf_path, rel_link, rel_base, num):
        self.model = optas.RobotModel(urdf_filename=str(urdf_path))
        self.ref = RobotModelTester(str(urdf_path))
        self.root = self.model.get_root_link()
        self.rel_link, self.rel_base, self.num = rel_link, rel_base, num

    def transform(self, q, link, base):
        """Reference transform of link in the base frame."""
        T = lambda name: self.ref.fkine(q, end=name).A
        return np.linalg.inv(T(base)) @ T(link)

    def jacobian(self, q, link, base, kind, h=1e-6):
        """Reference Jacobian of link in the base frame.

        Geometric parts are the world Jacobian rotated into the base frame. The
        analytical angular part is d(rpy of link w.r.t. base)/dq, by central differences.
        """
        J0 = self.ref.jacob0(q, end=link)
        J0 = np.pad(J0, ((0, 0), (0, len(q) - J0.shape[1])))  # joints after link
        R = self.ref.fkine(q, end=base).R.T
        rpy = lambda q: Rot.from_matrix(self.transform(q, link, base)[:3, :3]).as_euler(
            "xyz"
        )
        # Divide by the step actually taken: q + h is inexact for large q (joint0)
        steps = [(q + d, q - d) for d in h * np.eye(len(q))]
        J_rpy = np.column_stack(
            [wrap(rpy(a) - rpy(b)) / (a - b)[i] for i, (a, b) in enumerate(steps)]
        )
        lin, ang = R @ J0[:3], R @ J0[3:]
        return {
            "geometric": np.vstack([lin, ang]),
            "analytical": np.vstack([lin, J_rpy]),
            "linear": lin,
            "angular_geometric": ang,
            "angular_analytical": J_rpy,
        }[kind]


# The planar tester robot, and the KUKA LWR whose mixed joint axes catch
# rotation-order and axis mix-ups that a planar robot can't
ROBOTS = {
    "tester": Robot(TESTS_PATH / "tester_robot.urdf", "eff", "link1", NUM_RANDOM),
    "kuka_lwr": Robot(
        EXAMPLE_PATH / "robots" / "kuka_lwr" / "kuka_lwr.urdf",
        "end_effector",
        "lwr_arm_2_link",
        NUM_RANDOM // 5,
    ),
}

POSES = {
    "transform": lambda T: T,
    "position": lambda T: T[:3, 3],
    "rotation": lambda T: T[:3, :3],
    "quaternion": lambda T: Rot.from_matrix(T[:3, :3]).as_quat(),
    "rpy": lambda T: Rot.from_matrix(T[:3, :3]).as_euler("xyz"),
}
JACOBIANS = [
    "geometric",
    "analytical",
    "linear",
    "angular_geometric",
    "angular_analytical",
]
# Axis given by name, or as an (unnormalized) vector in the link frame
AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "vector": [1, 2, 3]}


@pytest.mark.parametrize("frame", ["global", "relative"])
@pytest.mark.parametrize("robot", ROBOTS.values(), ids=ROBOTS.keys())
class TestKinematics:
    """Compare get_global_link_<name> (every link, root base) and get_link_<name>
    (the robot's link/base pair), and their _function versions, with roboticstoolbox."""

    def _check(self, robot, frame, name, expected, *args):
        model = robot.model
        if frame == "global":
            pairs = [(link, robot.root) for link in model.link_names]
            get = lambda link, base, q: getattr(model, f"get_global_link_{name}")(
                link, q, *args
            )
            fun = lambda link, base, n: getattr(
                model, f"get_global_link_{name}_function"
            )(link, *args, n=n)
        else:
            pairs = [(robot.rel_link, robot.rel_base)]
            get = lambda link, base, q: getattr(model, f"get_link_{name}")(
                link, q, *args, base
            )
            fun = lambda link, base, n: getattr(model, f"get_link_{name}_function")(
                link, *args, base, n=n
            )

        def assert_close(out, exp):
            out, exp = np.asarray(out).flatten(), np.asarray(exp).flatten()
            if name == "rpy":  # e.g. a roll of pi or -pi
                out = exp + wrap(out - exp)
            # q and -q are the same quaternion
            assert isclose(out, exp, 1e-6) or (
                name == "quaternion" and isclose(out, -exp, 1e-6)
            )

        for link, base in pairs:
            for _ in range(robot.num // 10):
                n = np.random.randint(2, 10)
                q = model.get_random_joint_positions(n=n).toarray()
                out = fun(link, base, n)(q)
                for i in range(n):
                    exp = expected(q[:, i], link, base)
                    assert_close(get(link, base, q[:, i]).toarray(), exp)
                    out_i = out[i] if isinstance(out, list) else out[:, i]
                    assert_close(out_i.toarray(), exp)
            if link != base:  # the root's pose is constant, returned as DM
                q = optas.SX.sym("q", model.ndof)
                assert isinstance(get(link, base, q), optas.SX)

    @pytest.mark.parametrize("kind", POSES)
    def test_pose(self, robot, frame, kind):
        expected = lambda q, link, base: POSES[kind](robot.transform(q, link, base))
        self._check(robot, frame, kind, expected)

    @pytest.mark.parametrize("kind", JACOBIANS)
    def test_jacobian(self, robot, frame, kind):
        expected = lambda q, link, base: robot.jacobian(q, link, base, kind)
        self._check(robot, frame, f"{kind}_jacobian", expected)

    @pytest.mark.parametrize("axis", AXES)
    def test_axis(self, robot, frame, axis):
        a = np.array(AXES[axis]) / np.linalg.norm(AXES[axis])
        expected = lambda q, link, base: robot.transform(q, link, base)[:3, :3] @ a
        arg = axis if axis != "vector" else optas.DM(AXES[axis])
        self._check(robot, frame, "axis", expected, arg)


class TestRnea:
    # Setup path to tester robot URDF
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_path = os.path.join(cwd, "tester_robot_revolute.urdf")

    # Setup model (with no parameterized joints) and reference model
    model = optas.RobotModel(urdf_filename=urdf_path)
    tester_robot_model = RobotModelTester(urdf_path)

    def test_calculate_inverse_dynamics_symbol(self):
        q = optas.SX.sym("q", self.model.ndof)
        qd = optas.SX.sym("qd", self.model.ndof)
        qdd = optas.SX.sym("qdd", self.model.ndof)
        tau1 = self.model.rnea(q, qd, qdd)
        assert isinstance(tau1, optas.SX)

    def test_calculate_inverse_dynamics(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            qd = self.model.get_random_joint_positions().toarray().flatten()
            qdd = self.model.get_random_joint_positions().toarray().flatten()
            tau1 = self.model.rnea(q, qd, qdd)
            tau2 = self.tester_robot_model.rne(q, qd, qdd)
            assert isclose(tau1.toarray().flatten(), tau2)


class TestDualArm:
    optas_path = pathlib.Path(
        __file__
    ).parent.parent.resolve()  # path to current working directory
    examples_path = optas_path / "example"
    nextage_urdf_path = examples_path / "robots" / "nextage" / "nextage.urdf"

    model = optas.RobotModel(urdf_filename=str(nextage_urdf_path.absolute()))

    tester_robot_model = RobotModelTester(str(nextage_urdf_path.absolute()))

    def test_dual_arm(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [
                np.array(fk) for fk in self.tester_robot_model.fkine_all(q)
            ]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = self.model.get_global_link_transform(link_name, q).toarray()
                assert isclose(T, expected_T)
