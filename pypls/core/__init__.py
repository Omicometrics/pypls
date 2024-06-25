from .nipals import nipals
from .pls import pls_c, pls_vip, summary_pls
from .opls import (correct_fit,
                   correct_x_1d,
                   correct_x_2d,
                   summary_opls,
                   opls_vip)
from .cross_validation import (kfold_cv_opls,
                               kfold_cv_pls,
                               kfold_cv_pls_reg,
                               kfold_prediction)
from .scale_xy import scale_x_class, scale_x_reg


__all__ = [
    "nipals",
    "pls_c",
    "pls_vip",
    "correct_fit",
    "correct_x_1d",
    "correct_x_2d",
    "opls_vip",
    "kfold_cv_opls",
    "kfold_cv_pls",
    "kfold_cv_pls_reg",
    "kfold_prediction",
    "summary_opls",
    "summary_pls",
    "scale_x_class",
    "scale_x_reg"
]
