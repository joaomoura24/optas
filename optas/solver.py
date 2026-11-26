import os
import subprocess
import numpy as np
import casadi as cs
from dataclasses import dataclass
from scipy.interpolate import interp1d
from .optimization import Optimization
from .models import RobotModel
from .spatialmath import ArrayType, CasADiArrayType
from typing import Dict, Tuple, List, Optional

################################################################
# Helpers


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


@dataclass
class ViolatedConstraint:
    """! A constraint lower <= value <= upper that is violated (see CasADiSolver.violated_constraints).

    The arrays have one element per constraint element (column-major for matrices).
    """

    label: str
    lower: np.ndarray
    value: np.ndarray
    upper: np.ndarray

    @property
    def violation(self) -> np.ndarray:
        """! How far each element is outside its bounds (zero when inside)."""
        return np.maximum(
            np.maximum(self.lower - self.value, self.value - self.upper), 0
        )

    def __str__(self) -> str:
        return (
            f"{self.label}: max violation {self.violation.max():.3g}\n"
            f"  lower {self.lower}\n  value {self.value}\n  upper {self.upper}"
        )


class CasADiSolver:
    """! Solves an optimization problem with a CasADi QP (conic) or NLP solver plugin."""

    def __init__(self, optimization: Optimization, error_on_fail: bool = False):
        """! Constructor for the CasADiSolver class, call setup next.

        @param optimization The optimization problem created by calling the build method of the OptimizationBuilder class.
        @param error_on_fail When True, after solve() is called, if the solver did not converge then a RuntimeError is thrown. Default is False.
        @return An instance of the CasADiSolver class.
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

        ## Statistics of the last solve, fetched from the solver when first asked for.
        self._stats = None

        # Bounds of the constraint rows, set in setup
        self._bounds = None
        self._bounds_p_index = np.zeros(0, dtype=int)
        self._rows = None  # functions built on first use of violated_constraints
        self._cost_terms = None  # and evaluate_cost_terms

    def setup(self, solver_name: str, solver_options: Optional[Dict] = None):
        """! Setup the optimization solver.

        @param solver_name The name of the solver to be used: any QP (conic) or NLP solver plugin available in CasADi, e.g. qpoases, osqp, daqp, proxqp, qrqp, highs, ipopt, bonmin, sqpmethod. Mixed-integer problems need an NLP solver that supports them (e.g. bonmin).
        @param solver_options Options passed to the CasADi plugin (see https://web.casadi.org/docs/), e.g. {"qpsol": "qrqp"} for sqpmethod, {"daqp": {"eps_prox": 1e-6}} or {"proxqp": {"eps_abs": 1e-9}}.
        @return The CasADiSolver instance, so that setup can be chained after the constructor.
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

        # Constraints lbg(p) <= g(x, p) <= ubg(p): one row per equality or bound
        if self.opt.ncon > 0:
            problem["g"] = cs.cse(self.opt.con(x, p))
            lbg, ubg = self.opt.lbcon(p), self.opt.ubcon(p)
            self._bounds = cs.Function("bounds", [p], [lbg, ubg])
            # Evaluated here, and again on reset_parameters when the entries of p they
            # depend on change (e.g. joint limits given as parameters rarely do)
            self._lbg, self._ubg = self._bounds(self._p)
            if self.opt.np > 0:
                used = cs.which_depends(cs.vertcat(lbg, ubg), p, 1, False)
                self._bounds_p_index = np.flatnonzero(used)
            self._bounds_p = self._p[self._bounds_p_index]

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
        index = self._bounds_p_index
        if index.size > 0:
            if not np.array_equal(self._p[index], self._bounds_p):
                self._bounds_p = self._p[index]
                self._lbg, self._ubg = self._bounds(self._p)

    def _solve(self) -> CasADiArrayType:
        """! Solve the optimization problem using the CasADi interface.

        @return Solution to the problem.
        """
        solver_input = {"x0": self._x0, "p": self._p}
        if self.opt.ncon > 0:
            solver_input["lbg"] = self._lbg
            solver_input["ubg"] = self._ubg
        self._solution = self._solver(**solver_input)
        self._stats = None  # fetched from the solver when first asked for
        return self._solution["x"]

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

    def _vectors(
        self, x: Dict[str, ArrayType], p: Optional[Dict[str, ArrayType]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """! Vectorize x and p (when p is None, the parameters set by reset_parameters)."""
        xv = np.zeros(self.opt.nx)
        _pack(xv, self._x_layout, x)
        if p is None:
            return xv, self._p
        pv = np.zeros(self.opt.np)
        _pack(pv, self._p_layout, p)
        return xv, pv

    def violated_constraints(
        self,
        x: Dict[str, ArrayType],
        p: Optional[Dict[str, ArrayType]] = None,
        tol: float = 1e-6,
    ) -> List["ViolatedConstraint"]:
        """! Return the constraints that x violates, e.g. to see why a solve failed.

        A constraint is violated when any of its elements is outside
        [lower - tol, upper + tol].

        @param x The values for the decision variables, e.g. the dictionary returned by solve.
        @param p The values for the parameters. Default: the parameters set by reset_parameters.
        @param tol Tolerance on the bounds.
        @return List of ViolatedConstraint, in the order the constraints were added.
        """
        if self._rows is None:
            outputs = []
            for lower, expression, upper in self.opt.constraint_rows.values():
                ones = cs.DM.ones(*expression.shape)
                outputs += [
                    cs.vec(lower * ones),
                    cs.vec(expression),
                    cs.vec(upper * ones),
                ]
            self._rows = cs.Function("rows", [self.opt.x, self.opt.p], outputs)

        values = self._rows(*self._vectors(x, p))
        violated = []
        for i, label in enumerate(self.opt.constraint_rows):
            lower, value, upper = [
                np.asarray(v).flatten() for v in values[3 * i : 3 * i + 3]
            ]
            if np.any(value < lower - tol) or np.any(value > upper + tol):
                violated.append(ViolatedConstraint(label, lower, value, upper))
        return violated

    def evaluate_cost(
        self, x: Dict[str, ArrayType], p: Optional[Dict[str, ArrayType]] = None
    ) -> float:
        """! Evaluate the cost function.

        @param x The values for the decision variables.
        @param p The values for the parameters. Default: the parameters set by reset_parameters.
        @return The cost at x and p.
        """
        return float(self.opt.f(*self._vectors(x, p)))

    def evaluate_cost_terms(
        self, x: Dict[str, ArrayType], p: Optional[Dict[str, ArrayType]] = None
    ) -> Dict[str, float]:
        """! Evaluate each cost term.

        @param x The values for the decision variables.
        @param p The values for the parameters. Default: the parameters set by reset_parameters.
        @return Dictionary of the value of each cost term, by label.
        """
        if self._cost_terms is None:
            terms = self.opt.cost_terms
            self._cost_terms = cs.Function(
                "cost_terms", [self.opt.x, self.opt.p], list(terms.values())
            )
        values = self._cost_terms(*self._vectors(x, p))
        if not isinstance(values, (list, tuple)):
            values = [values]
        return {label: float(v) for label, v in zip(self.opt.cost_terms, values)}

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
        if self.opt.ncon > 0:
            solver_input["lbg"], solver_input["ubg"] = self._bounds(p)
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
