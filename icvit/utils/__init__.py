from icvit.utils.dist import get_world_size
from icvit.utils.optim import (
    clip_gradients,
    cosine_scheduler,
    get_params_groups,
    has_batchnorms,
    trunc_normal_,
)
