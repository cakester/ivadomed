import ivadomed.visualize as imed_visualize
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
import pytest
import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ivadomed.maths as imed_math
from PIL import Image
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir,  __tmp_dir__

@pytest.mark.parametrize('arr,minv,maxv,dtype,res', [([2],0.0,0.0,np.float32,0.0)])
def test_math(arr,minv,maxv,dtype,res):
    ret = imed_math.rescale_values_array(np.array(arr),minv,maxv,dtype)
    assert ret == np.array(res)

