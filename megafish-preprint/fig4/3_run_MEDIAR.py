# Setup the environment according to the instruction in
# https://github.com/Lee-Gihun/MEDIAR

import time
import torch

from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import EnsemblePredictor

local_dir = "/spo82/ana/"
sample_header = "012_SeqFISH_"
# change this to the group name of the input images
group_input = "hcst_mip_reg_mip"

img_nos = [1, 2, 3, 4, 5, 6]


def main():

    model_path1 = "/spo82/mediar/weight/from_phase1.pth"
    model_path2 = "/spo82/mediar/weight/from_phase2.pth"

    weights1 = torch.load(model_path1, map_location="cpu")
    weights2 = torch.load(model_path2, map_location="cpu")

    model_args = {
        "classes": 3,
        "decoder_channels": [1024, 512, 256, 128, 64],
        "decoder_pab_channels": 256,
        "encoder_name": 'mit_b5',
        "in_channels": 3
    }

    model1 = MEDIARFormer(**model_args)
    model1.load_state_dict(weights1, strict=False)
    model2 = MEDIARFormer(**model_args)
    model2.load_state_dict(weights2, strict=False)

    for i in img_nos:
        sample_name = sample_header + str(i)
        print("===== " + sample_name + " =====")

        input_path = local_dir + sample_header + \
            str(i) + "_tif/" + group_input + "/0/"
        output_path = local_dir + sample_header + \
            str(i) + "_tif/" + group_input + "_lbl/0/"

        st = time.time()
        predictor = EnsemblePredictor(
            model1, model2, "cuda:0", input_path, output_path, algo_params={"use_tta": False})
        _ = predictor.conduct_prediction()
        print("Time: ", time.time() - st)


if __name__ == "__main__":
    main()
