import casadi as cs
from functools import cached_property
from .models import Model
from .sx_container import SXContainer
from .spatialmath import CasADiArrayType
from typing import Dict, List, Optional


def derive_jacobian_function(
    name: str, fun: cs.Function, x: CasADiArrayType, p: CasADiArrayType
) -> cs.Function:
    """! Compute the Jacobian of a given function using automatic differentiation.

    @param name The function name.
    @param fun The CasADi function.
    @param x The variables of the function.
    @param p The parameters of the function.
    @return The Jacobian of the function wrt the variables x.
    """
    jac = cs.jacobian(fun(x, p), x)
    return cs.Function("d" + name, [x, p], [jac])


def derive_hessian_function(
    name: str, fun: cs.Function, x: CasADiArrayType, p: CasADiArrayType
) -> cs.Function:
    """! Compute the Hessian of a given function using automatic differentiation.

    @param name The function name.
    @param fun The CasADi function.
    @param x The variables of the function.
    @param p The parameters of the function.
    @return The Hessian of the function wrt the variables x.
    """
    jac = cs.jacobian(fun(x, p), x)
    hes = cs.jacobian(jac, x)
    return cs.Function("dd" + name, [x, p], [hes])


class Optimization:
    """! Optimization problem built by OptimizationBuilder.

            min f(x, p)
             x
                subject to
                            k(x, p) = M(p).x + c(p) >= 0   (linear)
                            a(x, p) = A(p).x + b(p) == 0   (linear)
                            g(x, p) >= 0
                            h(x, p) == 0

    The constraints are also given as rows lbcon(p) <= con(x, p) <= ubcon(p),
    the form the CasADi solvers take. When the cost is quadratic, f(x, p) =
    x'.P(p).x + x'.q(p) (plus a term that does not depend on x). The functions
    of derivatives (df, ddf, P, q, M, c, A, b) are only derived when first used.
    """

    def __init__(
        self,
        decision_variables: SXContainer,
        parameters: SXContainer,
        cost_terms: SXContainer,
        lin_eq_constraints: Optional[SXContainer] = None,
        lin_ineq_constraints: Optional[SXContainer] = None,
        eq_constraints: Optional[SXContainer] = None,
        ineq_constraints: Optional[SXContainer] = None,
        constraint_rows: Optional[Dict] = None,
    ):
        """! Initializer for the Optimization class.

        @param decision_variables SXContainer containing decision variables.
        @param parameters SXContainer containing parameters.
        @param cost_terms SXContainer containing cost terms.
        @param lin_eq_constraints SXContainer containing the linear equality constraints a(x, p) == 0.
        @param lin_ineq_constraints SXContainer containing the linear inequality constraints k(x, p) >= 0.
        @param eq_constraints SXContainer containing the equality constraints h(x, p) == 0.
        @param ineq_constraints SXContainer containing the inequality constraints g(x, p) >= 0.
        @param constraint_rows Dictionary of (lower, expression, upper) for each constraint, see specify_bounded_constraints. By default, one row per constraint in the containers above.
        @return Instance of the Optimization class.
        """

        ## A list of the task and robot models (set during build method in the OptimizationBuilder class)
        self.models = None

        ## SXContainer containing decision variables.
        self.decision_variables = decision_variables

        ## SXContainer containing parameters.
        self.parameters = parameters

        ## SXContainer containing cost terms.
        self.cost_terms = cost_terms

        ## SXContainer containing linear equality constraints.
        self.lin_eq_constraints = (
            SXContainer() if lin_eq_constraints is None else lin_eq_constraints
        )

        ## SXContainer containing linear inequality constraints.
        self.lin_ineq_constraints = (
            SXContainer() if lin_ineq_constraints is None else lin_ineq_constraints
        )

        ## SXContainer containing equality constraints.
        self.eq_constraints = (
            SXContainer() if eq_constraints is None else eq_constraints
        )

        ## SXContainer containing inequality constraints.
        self.ineq_constraints = (
            SXContainer() if ineq_constraints is None else ineq_constraints
        )

        ## Vectorized decision variables.
        self.x = decision_variables.vec()

        ## Vectorized parameters.
        self.p = parameters.vec()

        ## Number of decision variables.
        self.nx = decision_variables.numel()

        ## Number of parameters.
        self.np = parameters.numel()

        f = cs.sum1(cost_terms.vec())

        ## CasADi function that evaluates the objective function.
        self.f = cs.Function("f", [self.x, self.p], [f])

        ## True when the cost is quadratic in x (then P and q are available).
        self.quadratic_cost = bool(cs.is_quadratic(f, self.x))

        def function(name, container):
            return cs.Function(name, [self.x, self.p], [container.vec()])

        ## CasADi function that evaluates the linear inequality constraints k(x, p) >= 0.
        self.k = function("k", self.lin_ineq_constraints)

        ## Number of linear inequality constraints.
        self.nk = self.lin_ineq_constraints.numel()

        ## CasADi function that evaluates the linear equality constraints a(x, p) == 0.
        self.a = function("a", self.lin_eq_constraints)

        ## Number of linear equality constraints.
        self.na = self.lin_eq_constraints.numel()

        ## CasADi function that evaluates the inequality constraints g(x, p) >= 0.
        self.g = function("g", self.ineq_constraints)

        ## Number of inequality constraints.
        self.ng = self.ineq_constraints.numel()

        ## CasADi function that evaluates the equality constraints h(x, p) == 0.
        self.h = function("h", self.eq_constraints)

        ## Number of equality constraints.
        self.nh = self.eq_constraints.numel()

        if constraint_rows is None:
            constraint_rows = {}
            for container, upper in (
                (self.lin_ineq_constraints, cs.inf),
                (self.ineq_constraints, cs.inf),
                (self.lin_eq_constraints, 0.0),
                (self.eq_constraints, 0.0),
            ):
                for label, expression in container.items():
                    constraint_rows[label] = (0.0, expression, upper)

        self.specify_bounded_constraints(constraint_rows)

    @cached_property
    def df(self) -> cs.Function:
        """! Jacobian of the objective function."""
        return derive_jacobian_function("f", self.f, self.x, self.p)

    @cached_property
    def ddf(self) -> cs.Function:
        """! Hessian of the objective function."""
        return derive_hessian_function("f", self.f, self.x, self.p)

    @cached_property
    def P(self) -> Optional[cs.Function]:
        """! CasADi function P(p) of a quadratic cost x'.P(p).x + x'.q(p), None when the cost is not quadratic."""
        if not self.quadratic_cost:
            return None
        return cs.Function("P", [self.p], [0.5 * self.ddf(self.x, self.p)])

    @cached_property
    def q(self) -> Optional[cs.Function]:
        """! CasADi function q(p) of a quadratic cost x'.P(p).x + x'.q(p), None when the cost is not quadratic."""
        if not self.quadratic_cost:
            return None
        x_zero = cs.DM.zeros(self.nx)
        return cs.Function("q", [self.p], [cs.vec(self.df(x_zero, self.p))])

    @cached_property
    def M(self) -> cs.Function:
        """! CasADi function M(p) of the linear inequality constraints k(x, p) = M(p).x + c(p)."""
        return cs.Function("M", [self.p], [cs.jacobian(self.k(self.x, self.p), self.x)])

    @cached_property
    def c(self) -> cs.Function:
        """! CasADi function c(p) of the linear inequality constraints k(x, p) = M(p).x + c(p)."""
        return cs.Function("c", [self.p], [self.k(cs.DM.zeros(self.nx), self.p)])

    @cached_property
    def A(self) -> cs.Function:
        """! CasADi function A(p) of the linear equality constraints a(x, p) = A(p).x + b(p)."""
        return cs.Function("A", [self.p], [cs.jacobian(self.a(self.x, self.p), self.x)])

    @cached_property
    def b(self) -> cs.Function:
        """! CasADi function b(p) of the linear equality constraints a(x, p) = A(p).x + b(p)."""
        return cs.Function("b", [self.p], [self.a(cs.DM.zeros(self.nx), self.p)])

    def set_models(self, models: List[Model]) -> None:
        """! Specify the models in the optimization problem.

        @param models A list of models (i.e. instance of a sub-class of Model).
        """
        self.models = models

    def specify_bounded_constraints(self, rows: Dict) -> None:
        """! Specify all constraints as lbcon(p) <= con(x, p) <= ubcon(p).

        An equality or a bound constraint is one row (equal bounds for an equality).
        This is the form CasADi solvers take.

        @param rows Dictionary of (lower, expression, upper) for each constraint.
        """
        lo, con, hi = [], [], []
        for lower, expression, upper in rows.values():
            ones = cs.DM.ones(*expression.shape)  # scalar bounds apply to every element
            lo.append(cs.vec(lower * ones))
            con.append(cs.vec(expression))
            hi.append(cs.vec(upper * ones))
        lo, con, hi = [cs.vertcat(cs.SX(0, 1), *parts) for parts in (lo, con, hi)]

        ## Dictionary of (lower, expression, upper) for each constraint.
        self.constraint_rows = dict(rows)

        ## CasADi function that evaluates all constraints as rows con(x, p), with bounds lbcon(p) <= con(x, p) <= ubcon(p).
        self.con = cs.Function("con", [self.x, self.p], [con])

        ## CasADi function that evaluates the lower bound of con, lbcon(p).
        self.lbcon = cs.Function("lbcon", [self.p], [lo])

        ## CasADi function that evaluates the upper bound of con, ubcon(p).
        self.ubcon = cs.Function("ubcon", [self.p], [hi])

        ## Number of constraint rows in con.
        self.ncon = con.numel()

    def has_discrete_variables(self) -> bool:
        """! True when at least one decision variable is discrete."""
        return self.decision_variables.has_discrete_variables()
