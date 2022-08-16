import os, pathlib

import numpy as np
import medpy.io
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import monai.transforms as tf
from monai.utils import set_determinism
import torch

from settings import loader_settings


class Seg:
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return

    def process(self):
        inp_path = loader_settings["InputPath"]  # Path for the input
        out_path = loader_settings["OutputPath"]  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # Convert 'dat' to Tensor, or as appropriate for your model.
            ###########
            ### Replace this section with the call to your code.
            transform = tf.Compose(
                [
                    tf.NormalizeIntensityd(keys=["image"], channel_wise=True),
                    tf.CopyItemsd(keys=["image"], times=1, names=["flipped_image"]),
                    tf.Flipd(keys=["flipped_image"], spatial_axis=0),
                    tf.ConcatItemsd(keys=["image", "flipped_image"], name="image"),
                    tf.ToTensord(keys=["image"]),
                ]
            )
            device = torch.device("cuda:0")
            model = UNet(
                dimensions=3,
                in_channels=2,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.2,
            ).to(device)
            set_determinism(seed=0)
            model.load_state_dict(
                torch.load(
                    "best_metric_model.pth"
                )  # TODO copy .pth file to main working dir
            )
            model.eval()
            roi_size = (96,) * 3
            sw_batch_size = 4
            post_pred = tf.AsDiscrete(argmax=True, to_onehot=2)
            img = dat[0]  # squeeze batch dimension for now
            img_dict = {"image": img}  # build dict to feed to MONAI transforms
            img_dict = transform(img_dict)  # apply MONAI transforms
            img = img_dict["image"]  # extract tensor from dict
            img = img.unsqueeze(0)  # add batch dimension manually
            img = img.to(device)  # cast to device
            amp = True  # TODO toggle if AMP causes problems
            if amp:
                with torch.cuda.amp.autocast():
                    prediction = sliding_window_inference(
                        img, roi_size, sw_batch_size, model
                    )  # prediction has batch dim
            else:
                prediction = sliding_window_inference(
                    img, roi_size, sw_batch_size, model
                )  # prediction has batch dim
            prediction = post_pred(prediction[0])[1]
            dat = prediction
            dat = dat.cpu().detach().numpy()
            # mean_dat = np.mean(dat)
            # dat[dat > mean_dat] = 1
            # dat[dat <= mean_dat] = 0
            ###
            ###########
            # dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f"=== saving {out_filepath} from {fil} ===")
            medpy.io.save(dat, out_filepath, hdr=hdr)
        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(
        parents=True, exist_ok=True
    )
    Seg().process()
