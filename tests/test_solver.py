import os
import shutil
import optas
import pytest
import numpy as np
import casadi as cs

###########################################################################
# Helper methods
#


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(np.asarray(A, dtype=float).ravel(), np.ravel(B)).all()


###########################################################################
# Tests
#


class _SolverTester:
    @staticmethod
    def setup_builder(constraint=False):
        # Booth function
        # https://www.sfu.ca/~ssurjano/booth.html
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        T = 1
        builder = optas.OptimizationBuilder(T)
        x = builder.add_decision_variables("x")
        y = builder.add_decision_variables("y")
        a = builder.add_parameter("a")
        b = builder.add_parameter("b")
        f = (x + a * y - b) ** 2 + (2.0 * x + y - 5.0) ** 2
        builder.add_cost_term("f", f)
        if constraint:
            builder.add_bound_inequality_constraint("bnd", -1e9, x, 1e9)
        return builder

    @staticmethod
    def setup_optimization(constraint=False):
        builder = _SolverTester.setup_builder(constraint=constraint)
        return builder.build()


class TestSolverInterface(_SolverTester):
    @staticmethod
    def solve_and_check_solution(solver, solver_name=""):
        solver.reset_initial_seed({"x": 0, "y": 0})
        solver.reset_parameters({"a": 2.0, "b": 7.0})
        name = f"{type(solver)}"
        if solver_name:
            name += "-" + solver_name
        result = solver.solve()
        assert isclose(result["x"].toarray().flatten(), 1.0), name + " solver failed"
        assert isclose(result["y"].toarray().flatten(), 3.0), name + " solver failed"
        assert solver.did_solve(), f"did_solve returns false for {name}"

    def test_casadi_interface(self):
        # any solver plugin CasADi has, not a fixed list (daqp needs casadi >= 3.7)
        qp_solver_names = ["qpoases", "qrqp"] + ["daqp"] * cs.has_conic("daqp")
        nlp_solver_names = ["ipopt", "sqpmethod"]
        for solver_name in qp_solver_names + nlp_solver_names:
            opt = self.setup_optimization(constraint=True)
            solver = optas.CasADiSolver(opt).setup(solver_name)
            self.solve_and_check_solution(solver, solver_name)

    def test_casadi_unknown_solver(self):
        with pytest.raises(ValueError):
            optas.CasADiSolver(self.setup_optimization()).setup("no_such_solver")


def test_reset_layout():
    """reset_parameters/reset_initial_seed match SXContainer.dict2vec."""
    builder = optas.OptimizationBuilder(T=1)
    X = builder.add_decision_variables("X", 2, 3)
    y = builder.add_decision_variables("y")
    A = builder.add_parameter("A", 2, 3)
    b = builder.add_parameter("b")
    builder.add_cost_term("f", optas.sumsqr(X - A) + (y - b) ** 2)
    opt = builder.build()
    solver = optas.CasADiSolver(opt).setup("qpoases")

    A_val = np.arange(6.0).reshape(2, 3)  # column-major order matters
    for p in [
        {"A": A_val, "b": 7.0},
        {"A": optas.DM(A_val), "b": [7.0], "unknown": 1.0},  # unknown is ignored
        {"b": 7.0},  # missing labels are zero
    ]:
        solver.reset_parameters(p)
        assert isclose(solver.p.toarray(), opt.parameters.dict2vec(p).toarray())
        x0 = {"X": p.get("A", 0 * A_val)}
        solver.reset_initial_seed(x0)
        assert isclose(
            solver.x0.toarray(), opt.decision_variables.dict2vec(x0).toarray()
        )

    with pytest.raises(ValueError):
        solver.reset_parameters({"A": np.ones(5)})  # wrong size

    solver.reset_parameters({"A": A_val, "b": 7.0})
    solution = solver.solve()
    assert solution["X"].shape == (2, 3)
    assert isclose(solution["X"].toarray(), A_val)
    assert isclose(solution["y"].toarray(), 7.0)


class TestCodegen(_SolverTester):
    # Booth function with a = 2, b = 7 has its minimum at x = 1, y = 3
    options = {
        "qpoases": {"printLevel": "none"},
        "qrqp": {"print_iter": False, "print_header": False, "print_info": False},
        "daqp": {},
    }

    def setup_solver(self, solver_name):
        opt = self.setup_optimization(constraint=True)
        solver = optas.CasADiSolver(opt).setup(solver_name, self.options[solver_name])
        solver.reset_parameters({"a": 2.0, "b": 0.0})
        return solver

    def test_to_function(self):
        solver = self.setup_solver("qpoases")
        # b is an input, a keeps the value given to reset_parameters
        F = solver.to_function("booth", inputs=["b"], outputs=["y", "x"])
        assert F.name_in() == ["b"] and F.name_out() == ["y", "x"]
        y, x = F(7.0)
        assert isclose(x, 1.0) and isclose(y, 3.0)
        # default: every parameter in, every decision variable out
        F = solver.to_function("booth_all")
        assert F.name_in() == ["a", "b"] and F.name_out() == ["x", "y"]
        assert isclose(F(a=2.0, b=7.0)["y"], 3.0)

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
    @pytest.mark.parametrize("solver_name", ["qrqp"] + ["daqp"] * cs.has_conic("daqp"))
    def test_generate(self, solver_name, tmp_path):
        solver = self.setup_solver(solver_name)
        F = solver.to_function("booth", inputs=["a", "b"])
        lib = solver.generate(F, str(tmp_path / "gen"))
        assert os.path.exists(tmp_path / "gen" / "booth.h")
        G = cs.external("booth", lib)
        x, y = G(2.0, 7.0)
        assert isclose(x, 1.0) and isclose(y, 3.0)

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
    def test_generate_unsupported_solver(self, tmp_path):
        solver = self.setup_solver("qpoases")  # CasADi can't generate qpOASES
        with pytest.raises(RuntimeError):
            solver.generate(solver.to_function("booth"), str(tmp_path))


class TestBoundedConstraints:
    @staticmethod
    def build():
        # min ||x - t||^2  s.t.  x0 + x1 = t,  lo <= x <= hi
        builder = optas.OptimizationBuilder(T=1)
        x = builder.add_decision_variables("x", 2)
        lo = builder.add_parameter("lo", 2)
        hi = builder.add_parameter("hi", 2)
        t = builder.add_parameter("t")
        builder.add_cost_term("f", optas.sumsqr(x - t))
        builder.add_equality_constraint("sum", x[0] + x[1], t)
        builder.add_bound_inequality_constraint("box", lo, x, hi)
        return builder, x

    def test_rows(self):
        builder, x = self.build()
        builder.add_leq_inequality_constraint("leq", x[0], 10.0)
        builder.add_bound_inequality_constraint("x_dep", x[1] - 5.0, x[0], 5.0)
        opt = builder.build()
        # one row per equality and bound; a bound that uses x is two rows
        assert opt.ncon == 1 + 2 + 1 + 2
        assert opt.nv == 2 + 4 + 1 + 2  # v(x, p) >= 0 form, unchanged

    @pytest.mark.parametrize(
        "solver_name", ["qpoases", "qrqp"] + ["daqp"] * cs.has_conic("daqp")
    )
    def test_parameter_bounds(self, solver_name):
        options = TestCodegen.options[solver_name]
        solver = optas.CasADiSolver(self.build()[0].build()).setup(solver_name, options)
        p = {"lo": [-1.0, -1.0], "hi": [1.0, 1.0], "t": 1.0}
        solver.reset_parameters(p)
        assert isclose(solver.solve()["x"], [0.5, 0.5])
        # new bounds must take effect (they are only re-evaluated when they change)
        solver.reset_parameters({**p, "hi": [0.2, 1.0]})
        assert isclose(solver.solve()["x"], [0.2, 0.8])
        solver.reset_parameters(p)
        assert isclose(solver.solve()["x"], [0.5, 0.5])
        # and as inputs of to_function
        F = solver.to_function("F", inputs=["hi"])
        assert isclose(F([0.2, 1.0]), [0.2, 0.8])
