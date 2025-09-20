from ultrasphere_harmonics._ndim import harm_n_ndim_le


def test_harm_n_ndim_le() -> None:
    assert harm_n_ndim_le(0, e_ndim=1) == 0
    assert harm_n_ndim_le(1, e_ndim=1) == 1
    assert harm_n_ndim_le(2, e_ndim=1) == 2
    assert harm_n_ndim_le(3, e_ndim=1) == 2
    assert harm_n_ndim_le(9999, e_ndim=1) == 2

    assert harm_n_ndim_le(0, e_ndim=2) == 0
    assert harm_n_ndim_le(1, e_ndim=2) == 1
    assert harm_n_ndim_le(2, e_ndim=2) == 3
    assert harm_n_ndim_le(3, e_ndim=2) == 5
    assert harm_n_ndim_le(500, e_ndim=2) == 999

    assert harm_n_ndim_le(0, e_ndim=3) == 0
    assert harm_n_ndim_le(1, e_ndim=3) == 1
    assert harm_n_ndim_le(2, e_ndim=3) == 4
    assert harm_n_ndim_le(3, e_ndim=3) == 9
