import casadi as cs
import numpy as np
from typing import Dict, List, Optional, Union
from .robot_model import RobotModel
from .sx_container import SXContainer
from .optimization import UnconstrainedQP,\
    LinearConstrainedQP, \
    NonlinearConstrainedQP, \
    UnconstrainedOptimization, \
    LinearConstrainedOptimization, \
    NonlinearConstrainedOptimization

class OptimizationBuilder:

    """The OptimizationBuilder class is used to build an optimization problem."""

    def __init__(
            self,
            robots: Dict[str, RobotModel],
            T: Optional[int]=1,
            qderivs: Optional[List[int]]=[0]
    ):
        """Constructor for the OptimizationBuilder class.

        Parameters
        ----------

        robots : dict[str:pyinvk.robot_model.RobotModel]
            A dictionary containing the robot models in the
            scene. Each model should be indexed by a unique name for
            the robot. If you are interfacing with the ROS-PyBullet
            Interface then the robot name should be set to the same
            name as is given in the interface configuration file.

        T : int (default is 1)
            Number of time-steps in the trajectory to be treated as
            decicision variables. The value of T should be strictly
            positive and satisfy the inequality T > max(qderivs).

        qderivs : list[int] (default is [0])
            A list containing the derivative orders that should be
            included in the decision variables. For example, [0, 1]
            will include the joint configuration trajectory (i.e. a
            vector q with T time-steps), and the joint velocities
            (i.e. a vector qd with T-1 time-steps). All values in
            qderivs should be positive or zero.

        """

        # Input check
        assert min(qderivs) >= 0, "All values in qderivs should be positive or zero"
        dorder = max(qderivs)
        assert T >= 1, "T must be strictly positive"
        assert dorder >= 0, "dorder must be non-negative"
        assert T > dorder, f"T must be greater than {dorder}"

        # Set class attributes
        self.qderivs = qderivs
        self.dorder = dorder
        self.robots = robots

        # Setup decision variables
        self.decision_variables = SXContainer()
        for robot_name, robot in robots.items():
            for qderiv in qderivs:
                n = self.statename(robot_name, qderiv)
                self.decision_variables[n] = cs.SX.sym(n, robot.ndof, T-qderiv)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self.parameters = SXContainer()
        self.cost_terms = SXContainer()
        self.lin_constraints = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()

    @staticmethod
    def statename(robot_name: str, qderiv: int) -> str:
        """Returns the state name for a given time derivative of q.

        The state name is used to index the decision variables.

        Parameters
        ----------

        robot_name : str
            Name of the robot, as given in the robots parameter in the
            class constructor.

        qderiv : int
            The derivative order.

        Returns
        -------

        state_name : str
            A string in the format "{robot_name}/[D]q" where "[D]" is
            qderiv-number of "d"'s, e.g. if robot_name="robot" and
            qderiv=2, then the state name will be "robot/qdd".

        """
        return robot_name + '/' + 'd'*qderiv + 'q'

    def get_state(
            self,
            robot_name: str,
            t: int,
            qderiv: Optional[int]=0) -> cs.casadi.SX:
        """Return the configuration state for a given time deriviative of q.

        Parameters
        ----------

        robot_name : str
            Name of the robot, as given in the robots parameter in the
            class constructor.

        t : int
            Time step in the trajectory.

        qderiv : int
            The derivative order.

        Returns
        -------

        state : casadi.casadi.SX
            The state for a given robot, at a given time derivative,
            for a given time-step.

        """
        assert qderiv in self.qderivs, f"{qderiv=}, qderiv must be in {self.qderivs}"
        states = self.decision_variables[self.statename(robot_name, qderiv)]
        return states[:, t]

    def add_decision_variables(
            self,
            name: str,
            m: Optional[int]=1,
            n: Optional[int]=1) -> cs.casadi.SX:
        """Add a decision variable to the optimization problem.

        Parameters
        ----------

        name : str
            Name for the decision variable.

        m : int (default is 1)
            Number of rows in the decision variable array.

        n : int (default is 1)
            Number of columns in the decision variable array.

        Returns
        -------

        decision_variable : casadi.casadi.SX
            The decision variable SX array.
        """
        x = cs.SX.sym(name, m, n)
        self.decision_variables[name] = x
        return x

    def add_cost_term(self, name: str, cost_term: cs.casadi.SX) -> None:
        """Add a cost term to the optimization problem.

        When the optimization problem is built, the cost function is
        given by a sum of cost terms.

        Parameters
        ----------

        name : str
            Name for the cost term.

        cost_term : casadi.casadi.SX
            The cost term as a SX variable. Note, this must be scalar,
            i.e. with shape (1, 1).

        """
        assert cost_term.shape == (1, 1), "cost terms must be scalars"
        self.cost_terms[name] = cost_term

    def add_parameter(self, name: str, m: Optional[int]=1, n : Optional[int]=1) -> cs.casadi.SX:
        """Add a parameter to the optimization problem.

        Parameters
        ----------

        name : str
            Name for the parameter. Note, this name is used when
            referencing the parameter for the reset_parameters method
            in the Solver class.

        m : int (default is 1)
            Number of rows in the parameter array.

        n : int (default is 1)
            Number of columns in the parameter array.

        Returns
        -------

        param : casadi.casadi.SX
            The SX parameter array.

        """
        p = cs.SX.sym(name, m, n)
        self.parameters[name] = p
        return p

    def add_lin_constraint(
            self,
            name: str,
            lbc: Union[cs.casadi.SX, cs.casadi.DM],
            c: cs.casadi.SX,
            ubc: Union[cs.casadi.SX, cs.casadi.DM]) -> None:
        """Add linear constraint.

        Adds a constraint that is assumed to be linear in x (the
        decision variables). When the constraint is not linear, an
        AssertionError is thrown. Note, the constraint is defined as

            lbc <= c <= ubc.

        This is included in the linear constraint variable container
        as two constraints, i.e.

            c - lbc >= 0, and
            ubc - c >= 0.

        Parameters
        ----------

        name : str
            Name for the linear constraint.

        lbc : Union[cs.casadi.SX, cs.casadi.DM]
            Lower bound for constraint.

        c : Union[cs.casadi.SX, cs.casadi.DM]
            The constraint array.

        ubc : Union[cs.casadi.SX, cs.casadi.DM]
            Upper bound for the constraint.

        """
        x = self.decision_variables.vec()
        lb = c - lbc
        ub = ubc - c
        assert cs.is_linear(lb, x) and cs.is_linear(ub, x), "constraint not linear"
        self.lin_constraints[name+'_lb'] = lb
        self.lin_constraints[name+'_ub'] = ub

    def add_ineq_constraint(
            self,
            name: str,
            lbc: Union[cs.casadi.SX, cs.casadi.DM],
            c: cs.casadi.SX,
            ubc: Union[cs.casadi.SX, cs.casadi.DM]) -> None:
        """Adds an inequality constraint to the optimization problem.

        Note, the constraint is defined as

            lbc <= c <= ubc.

        This is included in the linear constraint variable container
        as two constraints, i.e.

            c - lbc >= 0, and
            ubc - c >= 0.

        If the constraint is linear then it is added as a linear
        constraint (i.e. it will be added to the lin_constraints
        attribute, otherwise it will be logged in the ineq_constraints
        attribute.

        Parameters
        ----------

        name : str
            Name for the constraint.

        lbc : Union[cs.casadi.SX, cs.casadi.DM]
            Lower bound for constraint.

        c : Union[cs.casadi.SX, cs.casadi.DM]
            The constraint array.

        ubc : Union[cs.casadi.SX, cs.casadi.DM]
            Upper bound for the constraint.

        """
        x = self.decision_variables.vec()
        lb = c - lbc
        ub = ubc - c
        if cs.is_linear(lb, x) and cs.is_linear(ub, x):
            print(f"[WARN] given constraint '{name}' is linear in x, adding as linear contraint")
            self.add_lin_constraint(name, lbc, c, ubc)
            return
        self.ineq_constraints[name+'_lb'] = lb
        self.ineq_constraints[name+'_ub'] = ub

    def add_eq_constraint(
            self,
            name: str,
            lhsc: Union[cs.casadi.SX, cs.casadi.DM],
            rhsc: Optional[Union[cs.casadi.SX, cs.casadi.DM]]=None) -> None:
        """Adds an equality constraint to the optimization problem.

        Note, the constraint is defined as lhsc == rhsc.

        If the constraint is linear then it is added as a linear
        constraint (i.e. it will be added to the lin_constraints
        attribute, otherwise it will be logged in the eq_constraints
        attribute.

        Parameters
        ----------

        name : str
            Name for the constraint.

        lhsc : Union[cs.casadi.SX, cs.casadi.DM]
            Left hand side of the equality constraint.

        rhsc : Union[cs.casadi.SX, cs.casadi.DM] (default is None)
            Right hand side of the equality constraint. This is
            optional, if it is None then it is assumed to be the zero
            array with the same shape as lhsc.

        """
        if rhsc is None:
            rhsc = cs.DM.zeros(*lhsc.shape)
        x = self.decision_variables.vec()
        eq = lhsc - rhsc
        if cs.is_linear(eq, x):
            print(f"[WARN] given constraint '{name}' is linear in x, adding as linear contraint")
            self.add_lin_constraint(name, lhsc, rhsc, lhsc) # (lhs=rhs) is equiv to (lhs <= rhs and rhs <= lhs)
            return
        self.eq_constraints[name] = eq

    def build(self) -> Union[UnconstrainedQP,\
                             LinearConstrainedQP, \
                             NonlinearConstrainedQP, \
                             UnconstrainedOptimization, \
                             LinearConstrainedOptimization, \
                             NonlinearConstrainedOptimization]:
        """Build the optimization problem.

        For the given 

        - decision variables (x),
        - parameters (p),
        - cost function (cost), and
        - constraints (k/g/h)

        the approrpriate optimization problem is built. The type of
        the optimization problem is chosen depending on the cost
        function and constraints. The available problem types are as
        follows. Note, 

        - dash "'" means transpose and full-stop "." means the dot
          product, or matrix-matrix/matrix-vector multiplication,
        - 0 is used to denote the zero array with an appropriate
          dimension,
        - equality constraints can be represented by inequality
          constraints, i.e. lhs == rhs is equivalent to lhs <= rhs and
          lhs >= rhs, and
        - the problem type determines the solvers that are available
          to solve the problem.

        UnconstrainedQP: 
        
                min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
                 x
        
            The problem is unconstrained, and has quadratic cost
            function - note, P and q are derived from the given cost
            function (you don't have to explicitly state P/q).

        LinearConstrainedQP:

                min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
                 x

                subject to k(x, p) = M(p).x + c(p) >= 0

            The problem is constrained by only linear constraints and
            has a quadratic cost function - note, P/M and q/c are
            derived from the given cost function and constraints (you
            don't have to explicitly state P/q/M/c).

        NonlinearConstrainedQP:

                min cost(x, p) where cost(x) = x'.P(p).x + x'.q
                 x

                subject to 

                    k(x, p) = M(p).x + c(p) >= 0,
                    g(x) == 0, and 
                    h(x) >= 0

            The problem is constrained by nonlinear constraints and
            has a quadratic cost function - note, P/M and q/c are
            derived from the given cost function and constraints (you
            don't have to explicitly state P/q/M/c).

        UnconstrainedOptimization:

                min cost(x, p)
                 x        

            The problem is unconstrained and the cost function is
            nonlinear in x.

        LinearConstrainedOptimization:

                min cost(x, p)
                 x

                subject to k(x, p) = M(p).x + c(p) >= 0

            The problem is constrained with linear constraints and has
            a nonlinear cost function in x.

        NonlinearConstrainedOptimization:

                min cost(x, p)
                 x

                subject to 

                    k(x, p) = M(p).x + c(p) >= 0,
                    g(x) == 0, and 
                    h(x) >= 0

            The problem is constrained by nonlinear constraints and
            has a nonlinear cost function.

        Returns
        -------

        opt_problem : Union[UnconstrainedQP, 
                            LinearConstrainedQP,
                            NonlinearConstrainedQP,
                            UnconstrainedOptimization,
                            LinearConstrainedOptimization,
                            NonlinearConstrainedOptimization]
            The optimization problem of either one of the above
            types. The problem type determines what costraints (and
            their type) are available and also the structure of the
            cost function.

        """

        # Get decision variables and parameters as SX column vectors
        x = self.decision_variables.vec()
        p = self.parameters.vec()

        # Helpful method
        def functionize(name, fun):

            # Setup function input
            fun_input = [x, p]

            # Function
            Fun = cs.Function(name, fun_input, [fun])

            # Jacobian
            jac = cs.jacobian(fun, x)
            Jac = cs.Function('d'+name, fun_input, [jac])

            # Hessian
            hess = cs.jacobian(jac, x)
            Hess = cs.Function('dd'+name, fun_input, [hess])

            return Fun, Jac, Hess

        # Get forward functions
        f = cs.sum1(self.cost_terms.vec())
        k = self.lin_constraints.vec()
        g = self.eq_constraints.vec()
        h = self.ineq_constraints.vec()

        # Setup optimization
        nlin = self.lin_constraints.numel()  # no. linear constraints
        nnlin = self.eq_constraints.numel() + self.ineq_constraints.numel()  # no. nonlin constraints
        if cs.is_quadratic(f, x):
            # True -> use QP formulation
            if nnlin > 0:
                opt = NonlinearConstrainedQP()
            elif nlin > 0:
                opt = LinearConstrainedQP()
            else:
                opt = UnconstrainedQP()
        else:
            # False -> use (nonlinear) Optimization formulation
            if nnlin > 0:
                opt = NonlinearConstrainedOptimization()
            elif nlin > 0:
                opt = LinearConstrainedOptimization()
            else:
                opt = UnconstrainedOptimization()

        # Setup constraints
        if nnlin > 0:
            opt.k, opt.dk, opt.ddk = functionize('k', k)
            opt.g, opt.dg, opt.ddg = functionize('g', g)
            opt.h, opt.dh, opt.ddh = functionize('h', h)
        if nlin > 0:
            opt.k, opt.dk, opt.ddk = functionize('k', k)

        # Setup cost function and other variables
        opt.f, opt.df, opt.ddf = functionize('f', f)
        opt.decision_variables = self.decision_variables
        opt.parameters = self.parameters
        opt.cost_terms = self.cost_terms

        return opt
