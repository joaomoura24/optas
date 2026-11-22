import pytest
import optas
from optas.sx_container import SXContainer


def test_add():
    a = SXContainer({"xa": optas.SX.sym("xa"), "ya": optas.SX.sym("ya")})
    b = SXContainer({"xb": optas.SX.sym("xb"), "yb": optas.SX.sym("yb")})
    assert (a + b).numel() == 4


def test_setitem():
    a = SXContainer({"xa": optas.SX.sym("xa"), "ya": optas.SX.sym("ya")})
    a["b"] = optas.SX.sym("b")  # this should not fail
    with pytest.raises(AssertionError):
        a["xa"] = None
    with pytest.raises(KeyError):
        a["xa"] = optas.SX.sym("xa")


def test_discrete():
    a = SXContainer({"x": optas.SX.sym("x", 2), "y": optas.SX.sym("y")})
    assert not a.has_discrete_variables()
    a.variable_is_discrete("x")
    assert a.has_discrete_variables()
    assert a.discrete() == [True, True, False]
    # discrete labels must not leak into other containers
    assert not SXContainer({"z": optas.SX.sym("z")}).has_discrete_variables()


def test_vec():
    a = SXContainer({"xa": optas.SX.sym("xa"), "ya": optas.SX.sym("ya")})
    assert a.vec().numel() == 2


def test_dict2vec_vec2dict():
    a = SXContainer({"xa": optas.SX.sym("xa"), "ya": optas.SX.sym("ya")})
    v = a.dict2vec(
        {"xa": 1.0, "ya": 2.0, "z": 100.0}
    )  # z should be ignored, i.e. no error thrown
    d = a.vec2dict(v)
    assert len(d) == 2
    assert "xa" in d
    assert "ya" in d
    assert d["xa"] == 1.0
    assert d["ya"] == 2.0


def test_zero():
    a = SXContainer({"xa": optas.SX.sym("xa"), "ya": optas.SX.sym("ya")})
    d = a.zero()
    assert len(d) == 2
    assert "xa" in d
    assert "ya" in d
    assert d["xa"].toarray().flatten()[0] == 0
    assert d["ya"].toarray().flatten()[0] == 0
