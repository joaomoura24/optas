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

tester_robot_model = RobotModelTester()

###########################################################################
# Helper methods
#


def random_vector(lo=-1, hi=1, n=3):
    return np.random.uniform(-lo, hi, size=(n,))


def isclose(A: np.ndarray, B: np.ndarray, atol=1e-08):
    return np.isclose(A, B, atol=atol).all()


def expected_link_transform(q, link, base):
    """Reference transform of link in the base frame, from roboticstoolbox."""
    T = lambda name: tester_robot_model.fkine(q, end=name).A
    return np.linalg.inv(T(base)) @ T(link)


def expected_link_jacobian(q, link, base, kind, h=1e-6):
    """Reference Jacobian of link in the base frame, from roboticstoolbox.

    Geometric parts are the world Jacobian rotated into the base frame. The
    analytical angular part is d(rpy of link w.r.t. base)/dq, by central differences.
    """
    J0 = tester_robot_model.jacob0(q, end=link)
    R = tester_robot_model.fkine(q, end=base).R.T
    rpy = lambda q: Rot.from_matrix(expected_link_transform(q, link, base)[:3, :3])
    dq = h * np.eye(len(q))
    J_rpy = np.column_stack(
        [
            (rpy(q + d).as_euler("xyz") - rpy(q - d).as_euler("xyz")) / (2 * h)
            for d in dq
        ]
    )
    lin, ang = R @ J0[:3], R @ J0[3:]
    return {
        "geometric": np.vstack([lin, ang]),
        "analytical": np.vstack([lin, J_rpy]),
        "linear": lin,
        "angular_geometric": ang,
        "angular_analytical": J_rpy,
    }[kind]


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

    def test_get_global_link_transform(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [np.array(fk) for fk in tester_robot_model.fkine_all(q)]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = self.model.get_global_link_transform(link_name, q).toarray()
                assert isclose(T, expected_T)

    def test_get_global_link_transform_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            for link_idx, link_name in enumerate(self.model.link_names):
                Tr = self.model.get_global_link_transform_function(link_name, n=n)
                Trs = Tr(q)

                for i in range(n):
                    qi = q[:, i].flatten()
                    expected_fkine = [
                        np.array(fk) for fk in tester_robot_model.fkine_all(qi)
                    ]
                    T_test = Trs[i].toarray()
                    T_expc = expected_fkine[link_idx + 1]
                    assert isclose(T_test, T_expc)

    def test_get_global_link_transform_function_numpy_output(self):
        link_name = self.model.link_names[-1]
        q = self.model.get_random_joint_positions().toarray()
        Tr = self.model.get_global_link_transform_function(link_name, numpy_output=True)
        T = Tr(q)
        assert isinstance(T, np.ndarray)

        q = optas.SX.sym("q", q.shape[0])
        with pytest.raises(AssertionError):
            T = Tr(q)

    # Relative-pose tests use eff w.r.t. link1. The base must not be adjacent: eff
    # w.r.t. link3 is a constant offset, which can't tell inv(T_B) @ T_L apart from
    # other compositions of the two transforms.
    rel_link, rel_base = "eff", "link1"
    rel_expected = {
        "transform": lambda T: T,
        "position": lambda T: T[:3, 3],
        "rotation": lambda T: T[:3, :3],
        "quaternion": lambda T: Rot.from_matrix(T[:3, :3]).as_quat(),
        "rpy": lambda T: Rot.from_matrix(T[:3, :3]).as_euler("xyz"),
    }

    def _check_link(self, kind, q, out):
        T = expected_link_transform(q, self.rel_link, self.rel_base)
        out, exp = np.asarray(out).flatten(), self.rel_expected[kind](T).flatten()
        if kind == "quaternion":  # q and -q are the same rotation
            assert isclose(out, exp) or isclose(out, -exp)
        else:
            assert isclose(out, exp)

    @pytest.mark.parametrize("kind", rel_expected)
    def test_get_link(self, kind):
        get = getattr(self.model, f"get_link_{kind}")
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            self._check_link(kind, q, get(self.rel_link, q, self.rel_base).toarray())

    @pytest.mark.parametrize("kind", rel_expected)
    def test_get_link_function(self, kind):
        for _ in range(NUM_RANDOM // 10):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()
            fun = getattr(self.model, f"get_link_{kind}_function")
            out = fun(self.rel_link, self.rel_base, n=n)(q)
            for i in range(n):
                out_i = out[i] if isinstance(out, list) else out[:, i]
                self._check_link(kind, q[:, i], out_i.toarray())

    def test_get_global_link_position(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [
                np.array(fk)[:3, 3].flatten() for fk in tester_robot_model.fkine_all(q)
            ]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = (
                    self.model.get_global_link_position(link_name, q)
                    .toarray()
                    .flatten()
                )
                assert isclose(T, expected_T)

    def test_get_global_link_position_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            for link_idx, link_name in enumerate(self.model.link_names):
                Tr = self.model.get_global_link_position_function(link_name, n=n)
                Trs = Tr(q)

                for i in range(n):
                    qi = q[:, i].flatten()
                    expected_fkine = [
                        np.array(fk)[:3, 3].flatten()
                        for fk in tester_robot_model.fkine_all(qi)
                    ]
                    T_test = Trs[:, i].toarray().flatten()
                    T_expc = expected_fkine[link_idx + 1]
                    assert isclose(T_test, T_expc)

    def test_get_global_link_rotation(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [np.array(fk) for fk in tester_robot_model.fkine_all(q)]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = self.model.get_global_link_rotation(link_name, q).toarray()
                assert isclose(T, expected_T[:3, :3])

    def test_get_global_link_rotation_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            for link_idx, link_name in enumerate(self.model.link_names):
                Tr = self.model.get_global_link_rotation_function(link_name, n=n)
                Trs = Tr(q)

                for i in range(n):
                    qi = q[:, i].flatten()
                    expected_fkine = [
                        np.array(fk) for fk in tester_robot_model.fkine_all(qi)
                    ]
                    T_test = Trs[i].toarray()
                    T_expc = expected_fkine[link_idx + 1]
                    assert isclose(T_test, T_expc[:3, :3])

    def test_get_global_link_quaternion(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [
                Rot.from_matrix(fk.R).as_quat()
                for fk in tester_robot_model.fkine_all(q)
            ]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = (
                    self.model.get_global_link_quaternion(link_name, q)
                    .toarray()
                    .flatten()
                )
                assert isclose(T, expected_T) or isclose(T, -expected_T)

    def test_get_global_link_quaternion_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            for link_idx, link_name in enumerate(self.model.link_names):
                Tr = self.model.get_global_link_quaternion_function(link_name, n=n)
                Trs = Tr(q)

                for i in range(n):
                    qi = q[:, i].flatten()
                    expected_fkine = [
                        Rot.from_matrix(fk.R).as_quat()
                        for fk in tester_robot_model.fkine_all(qi)
                    ]
                    T_test = Trs[:, i].toarray().flatten()
                    T_expc = expected_fkine[link_idx + 1]
                    assert isclose(T_test, T_expc) or isclose(T_test, -T_expc)

    def test_get_global_link_rpy(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            expected_fkine = [
                Rot.from_matrix(fk.R).as_euler("xyz")
                for fk in tester_robot_model.fkine_all(q)
            ]
            for link_name, expected_T in zip(self.model.link_names, expected_fkine[1:]):
                T = self.model.get_global_link_rpy(link_name, q).toarray().flatten()
                assert isclose(T, expected_T)

    def test_get_global_link_rpy_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            for link_idx, link_name in enumerate(self.model.link_names):
                Tr = self.model.get_global_link_rpy_function(link_name, n=n)
                Trs = Tr(q)

                for i in range(n):
                    qi = q[:, i].flatten()
                    expected_fkine = [
                        Rot.from_matrix(fk.R).as_euler("xyz")
                        for fk in tester_robot_model.fkine_all(qi)
                    ]
                    T_test = Trs[:, i].toarray().flatten()
                    T_expc = expected_fkine[link_idx + 1]
                    assert isclose(T_test, T_expc)

    def test_get_global_link_geometric_jacobian(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = self.model.get_global_link_geometric_jacobian("eff", q)
            J_expc = tester_robot_model.jacob0(q)
            assert isclose(J.toarray(), J_expc)

    def test_get_global_link_geometric_jacobian_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            J = self.model.get_global_link_geometric_jacobian_function("eff", n=n)
            Js = J(q)

            for i in range(n):
                qi = q[:, i].flatten()
                J_test = Js[i].toarray()
                J_expc = tester_robot_model.jacob0(qi)
                assert isclose(J_test, J_expc)

    def test_get_global_link_analytical_jacobian(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = self.model.get_global_link_analytical_jacobian("eff", q)
            _J_expc = tester_robot_model.jacob0_analytical(q)
            J_expc = np.zeros_like(_J_expc)
            J_expc[:3, :] = _J_expc[:3, :]
            J_expc[3:, :] = _J_expc[3:, :][::-1, :]
            assert isclose(J.toarray(), J_expc)

    def test_get_global_link_analytical_jacobian_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            J = self.model.get_global_link_analytical_jacobian_function("eff", n=n)
            Js = J(q)

            for i in range(n):
                qi = q[:, i].flatten()
                J_test = Js[i].toarray()
                _J_expc = tester_robot_model.jacob0_analytical(qi)
                J_expc = np.zeros_like(_J_expc)
                J_expc[:3, :] = _J_expc[:3, :]
                J_expc[3:, :] = _J_expc[3:, :][::-1, :]
                assert isclose(J_test, J_expc)

    rel_jacobian_kinds = [
        "geometric",
        "analytical",
        "linear",
        "angular_geometric",
        "angular_analytical",
    ]

    def _check_link_jacobian(self, kind, q, out):
        J = expected_link_jacobian(q, self.rel_link, self.rel_base, kind)
        assert isclose(np.asarray(out), J, atol=1e-6)

    @pytest.mark.parametrize("kind", rel_jacobian_kinds)
    def test_get_link_jacobian(self, kind):
        get = getattr(self.model, f"get_link_{kind}_jacobian")
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = get(self.rel_link, q, self.rel_base).toarray()
            self._check_link_jacobian(kind, q, J)

    @pytest.mark.parametrize("kind", rel_jacobian_kinds)
    def test_get_link_jacobian_function(self, kind):
        for _ in range(NUM_RANDOM // 10):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()
            fun = getattr(self.model, f"get_link_{kind}_jacobian_function")
            out = fun(self.rel_link, self.rel_base, n=n)(q)
            for i in range(n):
                self._check_link_jacobian(kind, q[:, i], out[i].toarray())

    def test_get_global_link_linear_jacobian(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = self.model.get_global_link_linear_jacobian("eff", q).toarray()
            J_expc = tester_robot_model.jacob0(q)[:3, :]
            assert isclose(J, J_expc)

    def test_get_global_link_linear_jacobian_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            J = self.model.get_global_link_linear_jacobian_function("eff", n=n)
            Js = J(q)

            for i in range(n):
                qi = q[:, i].flatten()
                J_test = Js[i].toarray()
                J_expc = tester_robot_model.jacob0(qi)[:3, :]
                assert isclose(J_test, J_expc)

    def test_get_global_link_angular_geometric_jacobian(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = self.model.get_global_link_angular_geometric_jacobian(
                "eff", q
            ).toarray()
            J_expc = tester_robot_model.jacob0(q)[3:, :]
            assert isclose(J, J_expc)

    def test_get_global_link_angular_geometric_jacobian_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            J = self.model.get_global_link_angular_geometric_jacobian_function(
                "eff", n=n
            )
            Js = J(q)

            for i in range(n):
                qi = q[:, i].flatten()
                J_test = Js[i].toarray()
                J_expc = tester_robot_model.jacob0(qi)[3:, :]
                assert isclose(J_test, J_expc)

    def test_get_global_link_angular_analytical_jacobian(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            J = self.model.get_global_link_angular_analytical_jacobian(
                "eff", q
            ).toarray()
            J_expc = tester_robot_model.jacob0_analytical(q)[3:, :][::-1, :]
            assert isclose(J, J_expc)

    def test_get_global_link_angular_analytical_jacobian_function(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(2, 10)
            q = self.model.get_random_joint_positions(n=n).toarray()

            J = self.model.get_global_link_angular_analytical_jacobian_function(
                "eff", n=n
            )
            Js = J(q)

            for i in range(n):
                qi = q[:, i].flatten()
                J_test = Js[i].toarray()
                J_expc = tester_robot_model.jacob0_analytical(qi)[3:, :][::-1, :]
                assert isclose(J_test, J_expc)

    def test_get_link_axis(self):
        q = optas.SX.sym("q", self.model.ndof)
        axis = self.model.get_link_axis("eff", q, "x", self.model.get_root_link())
        assert isinstance(axis, optas.SX)

    def test_get_link_axis_function(self):
        q = optas.SX.sym("q", self.model.ndof)
        axis = self.model.get_link_axis_function("eff", "x", self.model.get_root_link())
        assert isinstance(axis(q), optas.SX)

    def test_get_global_link_axis(self):
        q = optas.SX.sym("q", self.model.ndof)
        axis = self.model.get_global_link_axis("eff", q, "x")
        assert isinstance(axis, optas.SX)

    def test_get_global_link_axis_function(self):
        q = optas.SX.sym("q", self.model.ndof)
        axis = self.model.get_global_link_axis_function("eff", "x")
        assert isinstance(axis(q), optas.SX)


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
