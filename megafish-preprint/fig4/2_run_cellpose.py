# conda create -y -n cellpose -c rapidsai -c conda-forge -c nvidia rapids=24.06 python=3.11 cuda-version=11.8
import tifffile
from cellpose.io import imread
from cellpose import models, io
import torch
import shutil
import os
import re
from tqdm import tqdm


def natural_sort(list_to_sort):
    def _natural_keys(text):
        def _atoi(text):
            return int(text) if text.isdigit() else text
        return [_atoi(c) for c in re.split(r"(\d+)", text)]
    return sorted(list_to_sort, key=_natural_keys)


# check if gpu can be used
print(torch.cuda.is_available())

# model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
model = models.CellposeModel(model_type='nuclei', gpu=True)

root_dir = "/spo82/ana/012_SeqFISH_IF/240807/"
file_header = "012_SeqFISH_IF_"
sample_no = 1
group = "hcsti_mip_reg_skc_slc"
footer = "_cpl"  # CellPose Label

input_path = os.path.join(root_dir, file_header + str(sample_no) + "_tif",
                          group, "0")

output_dir = os.path.join(root_dir, file_header + str(sample_no) + "_tif",
                          group + footer, "0")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

files = os.listdir(input_path)
files = natural_sort(files)
for file in tqdm(files):
    if file.endswith(".tif"):
        img = imread(os.path.join(input_path, file))
        masks, flows, styles = model.eval(img, diameter=80, channels=[0, 0])
        output_name = file.replace(".tif", "_label.tiff")
        tifffile.imsave(os.path.join(output_dir, output_name), masks)
