import os
import subprocess
import numpy as np
import casadi as cs
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.sparse import csc_matrix
from .optimization import (
    Optimization,
    QuadraticCostUnconstrained,
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
    NonlinearCostUnconstrained,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
    MixedIntegerNonlinearCostNonlinearConstrained,
)
from .models import RobotModel
from .spatialmath import ArrayType, CasADiArrayType
from typing import Dict, Tuple, List, Union, Optional

## Optimization problem types with quadratic cost.
QP_COST = {
    QuadraticCostUnconstrained,
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
}


## Optimization problem types with nonlinear cost.
NL_COST = {
    NonlinearCostUnconstrained,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
    MixedIntegerNonlinearCostNonlinearConstrained,
}

## Optimization problem types that are unconstrained.
UNCONSTRAINED_OPT = {QuadraticCostUnconstrained, NonlinearCostUnconstrained}


## Optimization problem types that are constrained.
CONSTRAINED_OPT = {
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
    MixedIntegerNonlinearCostNonlinearConstrained,
}

MIXED_INTEGER_OPT = {
    MixedIntegerNonlinearCostNonlinearConstrained,
}

################################################################
# Solver base class


def _pack(buf: np.ndarray, layout: Dict, values: Dict[str, ArrayType]) -> None:
    """! Write values into buf in place, with the layout of an SXContainer.

    Labels missing from values are zero and unknown labels are ignored, as in
    SXContainer.dict2vec.
    """
    buf[:] = 0.0
    for label, value in values.items():
        if label in layout:
            sl, shape = layout[label]
            v = np.asarray(value, dtype=float).ravel(order="F")
            if v.size != sl.stop - sl.start:
                raise ValueError(
                    f"'{label}' has {v.size} elements, expected shape {shape}"
                )
            buf[sl] = v


def _unpack(vec: ArrayType, layout: Dict) -> Dict[str, cs.DM]:
    """! Split vec into a dictionary, with the layout of an SXContainer."""
    vec = cs.vec(vec if isinstance(vec, cs.DM) else cs.DM(vec))
    offsets = [sl.start for sl, _ in layout.values()] + [vec.numel()]
    out = {}
    for (label, (_, shape)), v in zip(layout.items(), cs.vertsplit(vec, offsets)):
        out[label] = v if shape[1] == 1 else cs.reshape(v, *shape)
    return out


class Solver(ABC):
    """! Base solver interface class"""

    def __init__(self, optimization: Optimization, error_on_fail: bool = False):
        """! Constructor for the base Solver class.

        @param optimization The optimization problem created by calling the build method of the OptimizationBuilder class.
        @param error_on_fail When True, after solve() is called, if the solver did not converge then a RuntimeError is thrown. Default is False.
        @return An instance of the Solver class.
        """

        ## Instance of the optimization problem.
        self.opt = optimization

        # Slice and shape of every decision variable and parameter, computed once
        self._x_layout = optimization.decision_variables.layout()
        self._p_layout = optimization.parameters.layout()

        # Initial seed and parameter vectors, written in place on every reset
        self._x0 = np.zeros(optimization.nx)
        self._p = np.zeros(optimization.np)

        ## When True, after solve() is called, if the solver did not converge then a RuntimeError is thrown.
        self._error_on_fail = error_on_fail

        ## Solution container
        self._solution = None

    @property
    def opt_type(self) -> type:
        """! Optimization type.

        @return The type of the optimization problem.
        """
        return type(self.opt)

    @abstractmethod
    def setup(self, *args, **kwargs):
        """! Setup solver, note this method must return self. This is an abstract method."""
        pass

    @property
    def x0(self) -> cs.DM:
        """! Initial guess for the optimization problem (set using reset_initial_seed)."""
        return cs.DM(self._x0)

    @property
    def p(self) -> cs.DM:
        """! Parameter vector (set using reset_parameters)."""
        return cs.DM(self._p)

    def reset_initial_seed(self, x0: Dict[str, ArrayType]) -> None:
        """! Reset initial seed for the optimization problem.

        @param x0 The initial seed.
        """
        _pack(self._x0, self._x_layout, x0)

    def reset_parameters(self, p: Dict[str, ArrayType]) -> None:
        """! Reset the parameters for the optimization problem.

        @param p Specifies the parameters.
        """
        _pack(self._p, self._p_layout, p)

    @abstractmethod
    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem and return the optimal decision variables as an array. This is an abstract method.

        @return The solution from the solver.
        """
        pass

    def solve(self) -> Dict:
        """! Solve the optimization problem.

        @return A dictionary containing the solution.
        """
        solution = _unpack(self._solve(), self._x_layout)
        p_dict = None  # parameter dictionary, only built if a model needs it

        if self._error_on_fail and (not self.did_solve()):
            raise RuntimeError("Solver failed!")

        # Add full model state to the solution dictionary
        for model in self.opt.models:
            for d in model.time_derivs:
                n_s = model.state_name(d)
                n_s_x = model.state_optimized_name(d)
                if isinstance(model, RobotModel):
                    if model.num_param_joints > 0:
                        n_s_p = model.state_parameter_name(d)
                        t = solution[n_s_x].shape[1]
                        solution[n_s] = cs.DM.zeros(model.dim, t)
                        solution[n_s][model.optimized_joint_indexes, :] = solution[
                            n_s_x
                        ]
                        if p_dict is None:
                            p_dict = _unpack(self._p, self._p_layout)
                        solution[n_s][model.parameter_joint_indexes, :] = p_dict[n_s_p]
                    else:
                        solution[n_s] = solution[n_s_x]
                else:
                    solution[n_s] = solution[n_s_x]

        return solution

    @abstractmethod
    def stats(self):
        """! Return stats from solver. The return type is specifc to the solver used. This is an abstract method.

        @return The statistics returned by the solver. This is specified to the solver used.
        """
        pass

    def violated_constraints(
        self, x: Dict[str, ArrayType], p: Dict[str, ArrayType]
    ) -> Tuple:
        """! Indicate the violated constraints.

        @param x The values for the decision variables.
        @param p The values for the parameters.
        @return Several lists that contain information regarding which constraints are violated.
        """
        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)

        @dataclass
        class ViolatedConstraint:
            label: str
            ctype: str
            diff: cs.DM
            pattern: cs.DM

            def __str__(self):
                return f"\n{self.label} [{self.ctype}]:\n{self.pattern}\n"

            def __repr__(self):
                info = str(self)
                max_width = max(len(line) for line in info.split("\n"))
                return "=" * max_width + info + "-" * max_width + "\n"

            @property
            def verbose_info(self):
                info = str(self)
                info += f"{self.diff}\n"
                return info

        lin_eq_violated_constraints = []
        for label, sx_var in self.opt.lin_eq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            lin_eq_violated_constraints.append(
                ViolatedConstraint(label, "lin_eq", diff, diff >= 0.0)
            )

        eq_violated_constraints = []
        for label, sx_var in self.opt.eq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            eq_violated_constraints.append(
                ViolatedConstraint(label, "eq", diff, diff >= 0.0)
            )

        lin_ineq_violated_constraints = []
        for label, sx_var in self.opt.lin_ineq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            lin_ineq_violated_constraints.append(
                ViolatedConstraint(label, "lin_ineq", diff, diff >= 0.0)
            )

        ineq_violated_constraints = []
        for label, sx_var in self.opt.ineq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            ineq_violated_constraints.append(
                ViolatedConstraint(label, "ineq", diff, diff >= 0.0)
            )

        return (
            lin_eq_violated_constraints,
            eq_violated_constraints,
            lin_ineq_violated_constraints,
            ineq_violated_constraints,
        )

    @staticmethod
    def interpolate(traj: cs.DM, T: float, **interp_args) -> interp1d:
        """! Interpolate a trajectory

        @param traj The trajectory to be interpolated where the columns correspond to the states over time.
        @param T The time duration of the trajectory.
        @return An interpolated function.
        """
        assert isinstance(
            traj, cs.DM
        ), f"traj is incorrect type, got '{type(traj)}', expected casadi.DM'"
        t = np.linspace(0, T, traj.shape[1])
        return interp1d(t, traj.toarray(), **interp_args)

    @abstractmethod
    def did_solve(self) -> bool:
        """! Returns true when the solver solved the previous problem, false otherwise. This is an abstract method.

        @return Result of whether the solver succeeded or not.
        """
        pass

    @abstractmethod
    def number_of_iterations(self) -> int:
        """! Returns the number of iterations required to solve the problem. This is an abstract method.

        @return Number of iterations.
        """
        pass

    def evaluate_cost(
        self, x: Dict[str, ArrayType], p: Dict[str, ArrayType]
    ) -> CasADiArrayType:
        """! Evaluates the cost function for given decision variables x and parameters p.

        @param x The values for the decision variables.
        @param p The values for the parameters.
        @return The cost that results from x and p.
        """
        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)
        return self.opt.f(x, p)

    def evaluate_cost_terms(
        self, x: Dict[str, ArrayType], p: Dict[str, ArrayType]
    ) -> List:
        """! Evaluates each cost term for given decision variables and parameters.

        @param x The values for the decision variables.
        @param p The values for the parameters.
        @return List corresponding to each cost term evalauted at x, p.
        """

        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)

        @dataclass
        class CostTerm:
            label: str
            cost: float

            def __str__(self):
                return f"\n{self.label}: {self.cost}"

            def __repr__(self):
                info = str(self)
                max_width = max(len(line) for line in info.split("\n"))
                return "=" * max_width + info + "-" * max_width + "\n"

        cost_terms = []
        for label, sx_var in self.opt.cost_terms.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            c = fun(x, p)
            cost_terms.append(c)

        return cost_terms


################################################################
# CasADi solvers (https://web.casadi.org/)


class CasADiSolver(Solver):
    """! This is a base class for CasADi solver interfaces."""

    def setup(self, solver_name: str, solver_options: Optional[Dict] = None):
        """! Setup the optimization solver.

        @param solver_name The name of the solver to be used: any QP (conic) or NLP solver plugin available in CasADi, e.g. qpoases, osqp, daqp, proxqp, qrqp, highs, ipopt, bonmin, sqpmethod. Mixed-integer problems need an NLP solver that supports them (e.g. bonmin).
        @param solver_options Solver options passed to the solver. For details, see
            - https://casadi.sourceforge.net/v3.0.0/api/html/d9/d37/group__qpsol.html, and
            - https://casadi.sourceforge.net/v2.0.0/api/html/d6/d07/classcasadi_1_1NlpSolver.html
        @return An instance of the CasADiSovler class.
        """
        # Setup problem
        x = self.opt.decision_variables.vec()
        p = self.opt.parameters.vec()

        # cse: remove repeated subexpressions (e.g. forward kinematics in Jacobians)
        problem = {
            "x": x,
            "p": p,
            "f": cs.cse(self.opt.f(x, p)),
        }

        # Setup constraints

        ## Lower bound on constraints.
        self._lbg = None

        ## Upper bound on constraints
        self._ubg = None

        if self.opt_type in CONSTRAINED_OPT:
            problem["g"] = cs.cse(self.opt.v(x, p))
            self._lbg = self.opt.lbv
            self._ubg = self.opt.ubv

        # Get solver interface
        is_qp = cs.has_conic(solver_name) and not cs.has_nlpsol(solver_name)
        if is_qp and not self.opt.has_discrete_variables():
            sol = cs.qpsol
        elif cs.has_nlpsol(solver_name):
            sol = cs.nlpsol
        else:
            raise ValueError(
                f"solver '{solver_name}' is not available in CasADi or does not support this problem type"
            )

        # Check for discrete variables
        solver_options = dict(solver_options or {})
        if self.opt.has_discrete_variables():
            solver_options["discrete"] = self.opt.decision_variables.discrete()

        # Initialize solver

        ## Instance of the CasADi solver.
        self._solver = sol("solver", solver_name, problem, solver_options)

        # Solver libraries that generated code links (e.g. daqp, or the QP solver of sqpmethod)
        self._solver_libs = {solver_name, solver_options.get("qpsol", solver_name)}

        return self

    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem using the CasADi interface.

        @return Solution to the problem.
        """
        solver_input = {"x0": self._x0, "p": self._p}
        if self.opt_type in CONSTRAINED_OPT:
            solver_input["lbg"] = self._lbg
            solver_input["ubg"] = self._ubg
        self._solution = self._solver(**solver_input)
        self._stats = None  # fetched from the solver when first asked for
        return self._solution["x"]

    def stats(self) -> Dict:
        """! Statistics relating to the previous call to solve.

        @return Dictionary containing the statistics.
        """
        if self._stats is None:
            self._stats = self._solver.stats()
            self._stats["solution"] = self._solution
        return self._stats

    def did_solve(self) -> bool:
        """! True when the solver succeeded, False otherwise.

        @return Boolean indicating success of solver.
        """
        return self.stats()["success"]

    def number_of_iterations(self) -> int:
        """! Number of iterations it took the solver to converge.

        @return Number of iterations.
        """
        return self.stats()["iter_count"]

    def to_function(
        self,
        name: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ) -> cs.Function:
        """! CasADi function that solves the problem: parameters in, decision variables out.

        Parameters that are not inputs keep the values last given to reset_parameters,
        and the initial seed is the one last given to reset_initial_seed. The function
        can be called from Python, or turned into C with generate().

        @param name Name of the function (a valid C identifier, for generate).
        @param inputs Parameter labels that become inputs, in order. Default: all parameters.
        @param outputs Decision variable labels to return, in order. Default: all decision variables.
        @return CasADi function whose inputs and outputs are named after the labels.
        """
        inputs = list(self._p_layout) if inputs is None else list(inputs)
        outputs = list(self._x_layout) if outputs is None else list(outputs)
        args = {label: cs.MX.sym(label, *self._p_layout[label][1]) for label in inputs}
        p = cs.vertcat(
            *[
                cs.vec(args[label]) if label in args else cs.DM(self._p[sl])
                for label, (sl, _) in self._p_layout.items()
            ]
        )
        solver_input = {"x0": self._x0, "p": p}
        if self.opt_type in CONSTRAINED_OPT:
            solver_input["lbg"] = self._lbg
            solver_input["ubg"] = self._ubg
        x = self._solver(**solver_input)["x"]
        res = [
            cs.reshape(x[self._x_layout[l][0]], *self._x_layout[l][1]) for l in outputs
        ]
        return cs.Function(name, list(args.values()), res, inputs, outputs)

    def generate(
        self, function: cs.Function, directory: str, compile: bool = True
    ) -> str:
        """! Generate C code for a function from to_function, and compile it.

        Writes <name>.c and <name>.h into directory and, when compile is True, builds
        lib<name>.so with gcc, linking the solver library shipped with CasADi (e.g.
        daqp or highs). Only solvers that CasADi can generate code for work, e.g.
        qrqp, daqp, highs, or sqpmethod using one of them as its qpsol; qpoases,
        proxqp and ipopt do not. The function returns non-zero when the solver fails
        (CasADi's QP solvers do by default, see their error_on_fail option).

        Load the library with casadi.external(name, path), or with dlopen from C/C++
        through the CasADi C API declared in the header.

        @param function Function returned by to_function.
        @param directory Output directory, created if needed.
        @param compile When False, only the C code and header are written.
        @return Path to the shared library, or to the C file when compile is False.
        """
        os.makedirs(directory, exist_ok=True)
        name = function.name()
        cg = cs.CodeGenerator(name + ".c", {"with_header": True})
        cg.add(function)
        c_file = cg.generate(os.path.join(directory, ""))
        if not compile:
            return c_file

        casadi_dir = os.path.dirname(cs.__file__)
        flags = [f"-I{os.path.join(casadi_dir, 'include')}"]
        for lib in sorted(self._solver_libs):
            if os.path.isdir(os.path.join(casadi_dir, "include", lib)):
                flags.append(f"-I{os.path.join(casadi_dir, 'include', lib)}")
            if os.path.exists(os.path.join(casadi_dir, f"lib{lib}.so")):
                flags += [f"-L{casadi_dir}", f"-l{lib}", f"-Wl,-rpath,{casadi_dir}"]
        so_file = os.path.join(directory, f"lib{name}.so")
        cmd = ["gcc", "-O3", "-fPIC", "-shared", c_file, "-o", so_file, *flags, "-lm"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"compiling {c_file} failed (can CasADi generate code for "
                f"{sorted(self._solver_libs)}?):\n{result.stderr}"
            )
        return so_file


################################################################
# OSQP solver (https://osqp.org/)


class OSQPSolver(Solver):
    """OSQP solver interface."""

    def setup(self, use_warm_start, settings: Optional[Dict] = None):
        """! Setup solver.

        @param use_warm_start When true, the initial seed x0 is used as a warm start. Default is True.
        @param settings Settings that are passed to OSQP. Default is {}.
        @return The instance of the solve (i.e. self).
        """
        assert self.opt_type in QP_COST, "OSQP cannot solve this type of problem"
        import osqp  # optional dependency: pip install 'pyoptas[solvers]'

        self._osqp = osqp
        self.use_warm_start = use_warm_start
        self._settings = dict(settings or {})
        self._reset_parameters()
        return self

    def reset_parameters(self, p: Dict[str, ArrayType]) -> None:
        """! Reset the parameters.

        @param p The values for the parameters.
        """
        super().reset_parameters(p)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """! Internal method to reset parameters."""
        self._setup_input = {
            **self._settings,
            "P": csc_matrix(2.0 * self.opt.P(self._p).toarray()),
            "q": self.opt.q(self._p).toarray().flatten(),
        }
        if self.opt_type in CONSTRAINED_OPT:
            # M x + c >= 0 and A x + b = 0, i.e. l <= [M; A] x <= u
            b = self.opt.b(self._p).toarray().flatten()
            self._setup_input["A"] = csc_matrix(
                cs.vertcat(self.opt.M(self._p), self.opt.A(self._p)).toarray()
            )
            self._setup_input["l"] = np.concatenate(
                [-self.opt.c(self._p).toarray().flatten(), -b]
            )
            self._setup_input["u"] = np.concatenate([np.full(self.opt.nk, np.inf), -b])

    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem using OSQP.

        @return The solution of the optimization problem.
        """

        # Setup solver
        self.m = self._osqp.OSQP()
        self.m.setup(**self._setup_input)

        # Warm start optimization
        if self.use_warm_start:
            self.m.warm_start(x=self._x0)

        # Solve problem
        self._solution = self.m.solve()

        return self._solution.x

    def stats(self):
        """! Statistics for the solution given by OSQP.

        @return Statistics returned by OSQP. See details at https://osqp.org/.
        """
        return self._solution

    def did_solve(self) -> bool:
        """! Returns True when the problem was solved.

        @return Boolean indicating if the solver converged.
        """
        return self._solution.info.status == "solved"

    def number_of_iterations(self) -> int:
        """! Number of iterations it took the solver to converge.

        @return Number of iterations.
        """
        return self._solution.info.iter


################################################################
# CVXOPT QP solver (https://cvxopt.org/)


class CVXOPTSolver(Solver):
    """! CVXOPT solver interface."""

    def setup(self, solver_settings: Optional[Dict] = None):
        """! Setup the cvxopt solver interface.

        @param solver_settings Settings passed to the CVXOPT solver.
        @return The instance of the solve (i.e. self).
        """
        assert self.opt_type in QP_COST, "CVXOPT cannot solve this problem"
        import cvxopt  # optional dependency: pip install 'pyoptas[solvers]'

        self._cvxopt = cvxopt

        ## Input to the solver
        self._solver_input = dict(solver_settings or {})

        self._reset_parameters()
        return self

    def reset_parameters(self, p: Dict[str, ArrayType]):
        """! Reset the parameters.

        @param p The values for the parameters.
        """
        super().reset_parameters(p)
        self._reset_parameters()

    def _reset_parameters(self):
        """! Internal method to reset parameters."""
        cvxopt = self._cvxopt
        self._solver_input["P"] = cvxopt.matrix(2.0 * self.opt.P(self._p).toarray())
        self._solver_input["q"] = cvxopt.matrix(self.opt.q(self._p).toarray().flatten())
        if self.opt_type in CONSTRAINED_OPT:
            if self.opt.nk > 0:
                self._solver_input["G"] = cvxopt.matrix(-self.opt.M(self._p).toarray())
                self._solver_input["h"] = cvxopt.matrix(
                    self.opt.c(self._p).toarray().flatten()
                )
            if self.opt.na > 0:
                self._solver_input["A"] = cvxopt.matrix(self.opt.A(self._p).toarray())
                self._solver_input["b"] = cvxopt.matrix(-self.opt.b(self._p).toarray())

    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem using CVXOPT.

        @return The solution of the optimization problem.
        """
        self._solution = self._cvxopt.solvers.qp(**self._solver_input)
        return self._solution["x"]

    def stats(self):
        """! Statistics relating to the previous call to solve.

        @return Dictionary containing the statistics.
        """
        return self._solution

    def did_solve(self):
        """! Returns True when the problem was solved.

        @return Boolean indicating if the solver converged.
        """
        return self._solution["status"] == "optimal"

    def number_of_iterations(self):
        """! Number of iterations it took the solver to converge.

        @return Number of iterations.
        """
        return self._solution["iterations"]


################################################################
# Scipy Minimize solvers (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)


class ScipyMinimizeSolver(Solver):
    """Scipy solver (scipy.optimize.minimize) interface."""

    ## Methods that require the Jacobian of the objective
    methods_req_jac = {
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    ## Methods that require the Hessian of the objective
    methods_req_hess = {
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    ## Methods that handle constrained optimization problems.
    methods_handle_constraints = {"COBYLA", "SLSQP", "trust-constr"}

    def setup(
        self,
        method: str = "SLSQP",
        tol: Union[None, float] = None,
        options: Union[None, Dict] = None,
    ):
        """Setup the Scipy solver.

        @param method Type of solver. Default is "SLSQP".
        @param tol Tolerance for termination. When tol is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s) equal to tol. For detailed control, use solver-specific options.
        @param options A dictionary of solver options.
        @return The instance of the solve (i.e. self).
        """

        # Input check
        if self.opt_type in CONSTRAINED_OPT and (
            method not in ScipyMinimizeSolver.methods_handle_constraints
        ):
            raise TypeError(
                f"optimization problem has constraints, the method '{method}' is not suitable"
            )

        # Setup class attributes

        ## Container for the statistics.
        self._stats = None

        ## Method name.
        self.method = method

        # Setup minimize input parameters

        ## Input to the minimize method
        self.minimize_input = {
            "fun": self.f,
            "method": method,
            "x0": self._x0.copy(),
        }

        if tol is not None:
            self.minimize_input["tol"] = tol

        if options is not None:
            self.minimize_input["options"] = options

        if method in ScipyMinimizeSolver.methods_req_jac:
            self.minimize_input["jac"] = self.jac

        if method in ScipyMinimizeSolver.methods_req_hess:
            self.minimize_input["hess"] = self.hess

        ## Constraints definition passed to the minimize method.
        self._constraints = {}
        if method in ScipyMinimizeSolver.methods_handle_constraints:
            if method != "trust-constr":
                if self.opt_type in CONSTRAINED_OPT:
                    self._constraints["constr"] = {
                        "type": "ineq",
                        "fun": self.v,
                        "jac": self.dv,
                    }
            else:
                if self.opt.nk:
                    self._constraints["k"] = LinearConstraint(
                        A=csc_matrix(self.opt.M(self._p).toarray()),
                        lb=-self.opt.c(self._p).toarray().flatten(),
                        ub=self.opt.inf * np.ones(self.opt.nk),
                    )

                if self.opt.na:
                    eq = -self.opt.b(self._p).toarray().flatten()
                    self._constraints["a"] = LinearConstraint(
                        A=csc_matrix(self.opt.A(self._p).toarray()),
                        lb=eq,
                        ub=eq,
                    )

                if self.opt.ng:
                    self._constraints["g"] = NonlinearConstraint(
                        fun=self.g,
                        lb=np.zeros(self.opt.ng),
                        ub=self.opt.inf * np.ones(self.opt.ng),
                        jac=self.dg,
                        hess=self.ddg,
                    )

                if self.opt.nh:
                    self._constraints["h"] = NonlinearConstraint(
                        fun=self.h,
                        lb=np.zeros(self.opt.nh),
                        ub=np.zeros(self.opt.nh),
                        jac=self.dh,
                        hess=self.ddh,
                    )

        return self

    def f(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return float(self.opt.f(x, self._p).toarray().flatten()[0])

    def jac(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.df(x, self._p).toarray().flatten()

    def hess(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.ddf(x, self._p).toarray()

    def v(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.v(x, self._p).toarray().flatten()

    def dv(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.dv(x, self._p).toarray()

    def g(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.g(x, self._p).toarray().flatten()

    def dg(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.dg(x, self._p).toarray()

    def ddg(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.ddg(x, self._p).toarray()

    def h(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.h(x, self._p).toarray().flatten()

    def dh(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.dh(x, self._p).toarray()

    def ddh(self, x: cs.np.ndarray) -> cs.np.ndarray:
        """! Internal method."""
        return self.opt.ddh(x, self._p).toarray()

    def reset_initial_seed(self, x0) -> None:
        """! Reset initial seed for the optimization problem.

        @param x0 The initial seed.
        """
        super().reset_initial_seed(x0)
        self.minimize_input["x0"] = self._x0.copy()

    def reset_parameters(self, p: Dict[str, ArrayType]):
        """! Reset the parameters.

        @param p The values for the parameters.
        """
        super().reset_parameters(p)
        if self.method == "trust-constr":
            if self.opt.nk:
                self._constraints["k"].A = csc_matrix(self.opt.M(self._p).toarray())
                self._constraints["k"].lb = -self.opt.c(self._p).toarray().flatten()
            if self.opt.na:
                eq = -self.opt.b(self._p).toarray().flatten()
                self._constraints["a"].A = csc_matrix(self.opt.A(self._p).toarray())
                self._constraints["a"].lb = eq
                self._constraints["a"].ub = eq
        if self._constraints:
            self.minimize_input["constraints"] = list(self._constraints.values())

    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem using Scipy.

        @return The solution of the optimization problem.
        """
        self._solution = minimize(**self.minimize_input)
        return self._solution.x

    def stats(self):
        """! Statistics relating to the previous call to solve.

        @return Dictionary containing the statistics.
        """
        return self._solution

    def did_solve(self):
        """! Returns True when the problem was solved.

        @return Boolean indicating if the solver converged.
        """
        return self._solution.success

    def number_of_iterations(self):
        """! Number of iterations it took the solver to converge.

        @return Number of iterations.
        """
        return self._solution.nit
