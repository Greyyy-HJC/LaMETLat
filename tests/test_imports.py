import lametlat
from lametlat import correlators, ground_state, plotting, utils


def test_top_level_import():
    assert lametlat.__version__ == "0.1.0"


def test_core_subpackages_import():
    assert correlators.bootstrap is not None
    assert correlators.jackknife is not None
    assert correlators.bin_data is not None
    assert correlators is not None
    assert plotting is not None
    assert ground_state is not None
    assert utils is not None
    assert correlators.read_pt2_h5 is not None
    assert correlators.read_pt3_h5 is not None
    assert correlators.get_qda_ratio_data is not None
    assert correlators.get_pt3_ratio_data is not None
    assert correlators.get_sum_data is not None
    assert correlators.get_fh_data is not None
    assert plotting.extrapolation_comparison_plot is not None
    assert plotting.pt3_ratio_plot is not None
    assert plotting.qda_ratio_plot is not None
    assert ground_state.general_prior is not None
    assert ground_state.pt2_re_fcn is not None
    assert ground_state.pt3_ratio_re_fcn is not None
    assert ground_state.pt3_ratio_im_fcn is not None
    assert ground_state.sum_re_fcn is not None
    assert ground_state.sum_im_fcn is not None
    assert ground_state.fh_re_fcn is not None
    assert ground_state.fh_im_fcn is not None
    assert ground_state.qda_re_fcn is not None
    assert ground_state.qda_im_fcn is not None
    assert ground_state.ff_ratio_fcn is not None
    assert ground_state.ff_sum_fcn is not None
    assert ground_state.ff_ratio_two_state_fit is not None
    assert ground_state.ff_sum_two_state_fit is not None
    assert ground_state.ff_joint_two_state_fit is not None
