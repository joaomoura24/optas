# Changelog

Changes in this version of OpTaS compared with the original [cmower/optas](https://github.com/cmower/optas).

## Unreleased

### CasADi is the only solver back end

- `CasADiSolver` is the only solver. `ScipyMinimizeSolver`, `OSQPSolver` and `CVXOPTSolver` are removed, along with the `solvers` extra. Use a CasADi plugin instead (see the README Solvers section), e.g. `setup("ipopt")` instead of scipy's SLSQP, and `setup("qpoases")` or `setup("daqp")` instead of OSQP/CVXOPT.
- The abstract `Solver` base class is merged into `CasADiSolver`; `opt_type` is removed.
- The seven `Optimization` subclasses (`QuadraticCostUnconstrained`, ..., `MixedIntegerNonlinearCostNonlinearConstrained`) are replaced by one `Optimization` class. Use `opt.quadratic_cost`, `opt.ncon` and `opt.has_discrete_variables()` instead of the class type. On a problem without constraints, `k`, `a`, `g` and `h` are now functions with an empty output (they were `None`).
- `Optimization` no longer has the stacked `v(x, p) >= 0` form (`specify_v`, `v`, `dv`, `ddv`, `lbv`, `ubv`, `nv`), the constraint derivatives `dg`, `ddg`, `dh`, `ddh`, the constant bounds `lbk`/`ubk`/`lba`/`uba`/`lbg`/`ubg`/`lbh`/`ubh`, or the helpers `vertcon` and `derive_jacobian_and_hessian_functions`; only the removed solvers used them. All constraints are given as rows `lbcon(p) <= con(x, p) <= ubcon(p)`. `df`, `ddf`, `P`, `q`, `M`, `c`, `A` and `b` are derived on first use; `P` and `q` are `None` when the cost is not quadratic.
- `violated_constraints(x, p=None, tol=1e-6)` returns a list of the violated constraints only (`ViolatedConstraint` with `label`, `lower`, `value`, `upper` and `violation`), instead of four lists of every constraint; the old version also marked equalities as violated whenever they were negative.
- `evaluate_cost_terms` returns `{label: value}` and `evaluate_cost` a float; in both, `p` defaults to the parameters set by `reset_parameters`.

### Examples

- The ICRA 2023 paper experiments (`example/experiment1.py`, `experiment2.py`) and their EXOTica files are removed.
- The examples use CasADi solvers only. `pushing.py` had the pusher angle bounds swapped and the wrong sign on the normal force; both are fixed.
