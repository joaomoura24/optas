import optas
from optas.sx_container import SXContainer
import numpy as np

NUM_RANDOM = 100

###########################################################################
# Helper methods
#


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B).all()


###########################################################################
# Tests
#


def test_derive_jacobian_and_hessian():
    name = "test"
    x = optas.SX.sym("x", 2)
    p = optas.SX.sym("p", 2)

    f = p[0] * x[0] ** 2 + p[1] * x[1] ** 3
    fun = optas.Function("fun", [x, p], [f])

    Jac = optas.optimization.derive_jacobian_function(name, fun, x, p)
    Hes = optas.optimization.derive_hessian_function(name, fun, x, p)

    jac_known = optas.horzcat(2.0 * p[0] * x[0], 3.0 * p[1] * x[1] ** 2)
    Jac_known = optas.Function("test_jac_known", [x, p], [jac_known])

    hes_known = optas.diag(optas.vertcat(2.0 * p[0], 6.0 * p[1] * x[1]))
    Hes_known = optas.Function("test_hes_known", [x, p], [hes_known])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(2,))
        p = np.random.uniform(-10, 10, size=(2,))
        assert isclose(Jac(x, p).toarray(), Jac_known(x, p).toarray())
        assert isclose(Hes(x, p).toarray(), Hes_known(x, p).toarray())


def test_optimization():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    models = [
        optas.TaskModel("test_model_1", 2),
        optas.TaskModel("test_model_2", 3),
    ]

    opt = optas.optimization.Optimization(dv, pr, ct)
    assert opt.quadratic_cost is True
    assert opt.ncon == 0

    opt.set_models(models)
    assert opt.models == models

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostUnconstrained():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    opt = optas.optimization.Optimization(dv, pr, ct)
    assert opt.quadratic_cost is True
    assert opt.ncon == 0

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostLinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    opt = optas.optimization.Optimization(dv, pr, ct, lec, lic)
    assert opt.quadratic_cost is True
    assert opt.ncon == 3

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 0,
        "nh": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostNonlinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    ec = SXContainer()
    ec["e1"] = x[3] ** 2 - x[1] * x[2]
    ec["e2"] = x[0] - 2.0 * x[1] * x[4]

    ic = SXContainer()
    ic["ic"] = x[1] + x[2] * x[0]

    opt = optas.optimization.Optimization(dv, pr, ct, lec, lic, ec, ic)
    assert opt.quadratic_cost is True
    assert opt.ncon == 6

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def g_known(x, p):
        return np.array([x[1] + x[2] * x[0]])

    def h_known(x, p):
        return np.array(
            [
                x[3] ** 2 - x[1] * x[2],
                x[0] - 2.0 * x[1] * x[4],
            ]
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        assert isclose(opt.g(x, p).toarray().flatten(), g_known(x, p))
        assert isclose(opt.h(x, p).toarray().flatten(), h_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 1,
        "nh": 2,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostUnconstrained():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    opt = optas.optimization.Optimization(dv, pr, ct)
    assert opt.quadratic_cost is False
    assert opt.P is None and opt.q is None
    assert opt.ncon == 0

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostLinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    opt = optas.optimization.Optimization(dv, pr, ct, lec, lic)
    assert opt.quadratic_cost is False
    assert opt.P is None and opt.q is None
    assert opt.ncon == 3

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 0,
        "nh": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostNonlinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    ec = SXContainer()
    ec["e1"] = x[3] ** 2 - x[1] * x[2]
    ec["e2"] = x[0] - 2.0 * x[1] * x[4]

    ic = SXContainer()
    ic["ic"] = x[1] + x[2] * x[0]

    opt = optas.optimization.Optimization(dv, pr, ct, lec, lic, ec, ic)
    assert opt.quadratic_cost is False
    assert opt.P is None and opt.q is None
    assert opt.ncon == 6

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def g_known(x, p):
        return np.array([x[1] + x[2] * x[0]])

    def h_known(x, p):
        return np.array(
            [
                x[3] ** 2 - x[1] * x[2],
                x[0] - 2.0 * x[1] * x[4],
            ]
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        assert isclose(opt.g(x, p).toarray().flatten(), g_known(x, p))
        assert isclose(opt.h(x, p).toarray().flatten(), h_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 1,
        "nh": 2,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_MixedIntegerNonlinearCostNonlinearConstrained():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    dv["xd"] = optas.SX.sym("xd", 2)
    dv.variable_is_discrete("xd")
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    ec = SXContainer()
    ec["e1"] = x[3] ** 2 - x[1] * x[2]
    ec["e2"] = x[0] - 2.0 * x[1] * x[4]

    ic = SXContainer()
    ic["ic"] = x[1] + x[2] * x[0]

    opt = optas.optimization.Optimization(dv, pr, ct, lec, lic, ec, ic)
    assert opt.quadratic_cost is False
    assert opt.P is None and opt.q is None
    assert opt.ncon == 6

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def g_known(x, p):
        return np.array([x[1] + x[2] * x[0]])

    def h_known(x, p):
        return np.array(
            [
                x[3] ** 2 - x[1] * x[2],
                x[0] - 2.0 * x[1] * x[4],
            ]
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(8,)).tolist()
        for idx, discrete in enumerate(dv.discrete()):
            if discrete:
                x[idx] = np.random.randint(0, 1)

        x = np.asarray(x)

        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        assert isclose(opt.g(x, p).toarray().flatten(), g_known(x, p))
        assert isclose(opt.h(x, p).toarray().flatten(), h_known(x, p))

    attr_exp_value_map = {
        "nx": 8,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 1,
        "nh": 2,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_default_constraint_rows():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 3)
    x = dv.vec()

    ct = SXContainer()
    ct["c"] = optas.sumsqr(x)

    lec, lic, ec, ic = SXContainer(), SXContainer(), SXContainer(), SXContainer()
    lec["lec"] = x[0] - 1.0
    lic["lic"] = x[1]
    ec["ec"] = x[0] * x[2]
    ic["ic"] = x[1:] ** 2

    opt = optas.optimization.Optimization(dv, SXContainer(), ct, lec, lic, ec, ic)

    # one row per constraint: inequalities in [0, inf], equalities in [0, 0]
    assert list(opt.constraint_rows) == ["lic", "ic", "lec", "ec"]
    assert opt.ncon == 5
    x0 = [2.0, 3.0, 4.0]
    p0 = optas.DM.zeros(0)
    assert isclose(opt.con(x0, p0).toarray().flatten(), [3.0, 9.0, 16.0, 1.0, 8.0])
    assert isclose(opt.lbcon(p0).toarray().flatten(), [0.0] * 5)
    assert isclose(opt.ubcon(p0).toarray().flatten(), [np.inf] * 3 + [0.0] * 2)

    # rows given explicitly (as the builder does) replace the default ones
    rows = {"box": (-1.0, x, 1.0)}
    opt = optas.optimization.Optimization(dv, SXContainer(), ct, constraint_rows=rows)
    assert opt.ncon == 3
    assert isclose(opt.lbcon(p0).toarray().flatten(), [-1.0] * 3)
    assert isclose(opt.ubcon(p0).toarray().flatten(), [1.0] * 3)
