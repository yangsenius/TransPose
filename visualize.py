# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2

keypoint_name = {
    0: "nose",
    1: "eye(l)",
    2: "eye(r)",
    3: "ear(l)",
    4: "ear(r)",
    5: "sho.(l)",
    6: "sho.(r)",
    7: "elb.(l)",
    8: "elb.(r)",
    9: "wri.(l)",
    10: "wri.(r)",
    11: "hip(l)",
    12: "hip(r)",
    13: "kne.(l)",
    14: "kne.(r)",
    15: "ank.(l)",
    16: "ank.(r)",
    17: "random",
    18: "random",
}


class plt_config:
    def __init__(self, dataset_name):
        assert dataset_name == "coco", "{} dataset is not supported".format(
            dataset_name
        )
        self.n_kpt = 17
        # edge , color
        self.EDGES = [
            ([15, 13], [255, 0, 0]),  # l_ankle -> l_knee
            ([13, 11], [155, 85, 0]),  # l_knee -> l_hip
            ([11, 5], [155, 85, 0]),  # l_hip -> l_shoulder
            ([12, 14], [0, 0, 255]),  # r_hip -> r_knee
            ([14, 16], [17, 25, 10]),  # r_knee -> r_ankle
            ([12, 6], [0, 0, 255]),  # r_hip  -> r_shoulder
            ([3, 1], [0, 255, 0]),  # l_ear -> l_eye
            ([1, 2], [0, 255, 5]),  # l_eye -> r_eye
            ([1, 0], [0, 255, 170]),  # l_eye -> nose
            ([0, 2], [0, 255, 25]),  # nose -> r_eye
            ([2, 4], [0, 17, 255]),  # r_eye -> r_ear
            ([9, 7], [0, 220, 0]),  # l_wrist -> l_elbow
            ([7, 5], [0, 220, 0]),  # l_elbow -> l_shoulder
            ([5, 6], [125, 125, 155]),  # l_shoulder -> r_shoulder
            ([6, 8], [25, 0, 55]),  # r_shoulder -> r_elbow
            ([8, 10], [25, 0, 255]),
        ]  # r_elbow -> r_wrist


def plot_poses(
    img, skeletons, config=plt_config("coco"), save_path=None, dataset_name="coco"
):

    cmap = matplotlib.cm.get_cmap("hsv")
    canvas = img.copy()
    n_kpt = config.n_kpt
    for i in range(n_kpt):
        rgba = np.array(cmap(1 - i / n_kpt - 1.0 / n_kpt * 2))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            if len(skeletons[j][i]) > 2 and skeletons[j][i, 2] > 0:
                cv2.circle(
                    canvas,
                    tuple(skeletons[j][i, 0:2].astype("int32")),
                    3,
                    (255, 255, 255),
                    thickness=-1,
                )

    stickwidth = 2
    for i in range(len(config.EDGES)):
        for j in range(len(skeletons)):
            edge = config.EDGES[i][0]
            color = config.EDGES[i][1]
            if len(skeletons[j][edge[0]]) > 2:
                if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                    continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def update_config(cfg, yamlfilename):
    cfg.defrost()
    cfg.merge_from_file(yamlfilename)
    cfg.TEST.MODEL_FILE = osp.join(cfg.DATA_DIR, cfg.TEST.MODEL_FILE)
    cfg.freeze()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def inspect_atten_map_by_locations(
    image,
    model,
    query_locations,
    model_name="transposer",
    mode="dependency",
    threshold=None,
    device=torch.device("cuda"),
    kpt_color="white",
    img_name="image",
    save_img=False,
):
    r"""
    Visualize the attention maps in all of attention layers
    Args:
        image: shape -> [3, h, w]; type -> torch.Tensor;
        model: a pretrained model; type -> torch.nn.Module
        query_locations: shape -> [K,2]: type -> np.array
        mode: 'dependency' or 'affect'
        threshold: Default: None. If using it, recommend to be 0.01
    """

    assert mode in ["dependency", "affect"]
    inputs = torch.cat([image.to(device)]).unsqueeze(0)
    features = []
    global_enc_atten_maps = []
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    img_vis = image * std + mean
    img_vis = img_vis.permute(1, 2, 0).detach().cpu().numpy()
    img_vis_kpts = img_vis.copy()
    img_vis_kpts = plot_poses(img_vis_kpts, [query_locations])

    feature_hooks = [
        model.reduce.register_forward_hook(
            lambda self, input, output1: features.append(output1)
        )
    ]

    atten_maps_hooks = [
        model.global_encoder.layers[i].self_attn.register_forward_hook(
            lambda self, input, output: global_enc_atten_maps.append(output[1])
        )
        for i in range(len(model.global_encoder.layers))
    ]

    with torch.no_grad():
        outputs = model(inputs)
        del outputs
        for h in feature_hooks:
            h.remove()
        for h in atten_maps_hooks:
            h.remove()

    shape = features[0].shape[-2:]
    enc_atten_maps_hwhw = []
    for atten_map in global_enc_atten_maps:
        atten_map = atten_map.reshape(shape + shape)
        enc_atten_maps_hwhw.append(atten_map)

    attn_layers_num = len(enc_atten_maps_hwhw)
    down_rate = img_vis_kpts.shape[0] // shape[0]
    # query locations are at the coordinate frame of original image
    attn_map_pos = query_locations / down_rate

    # random pos
    x1 = img_vis_kpts.shape[1] * torch.rand(1)
    y1 = img_vis_kpts.shape[0] * torch.rand(1)
    x2 = img_vis_kpts.shape[1] * torch.rand(1)
    y2 = img_vis_kpts.shape[0] * torch.rand(1)
    random_pt_1 = [x1 / down_rate, y1 / down_rate]
    random_pt_2 = [x2 / down_rate, y2 / down_rate]
    attn_map_pos = attn_map_pos.tolist()
    attn_map_pos.append(random_pt_1)
    attn_map_pos.append(random_pt_2)

    fig, axs = plt.subplots(attn_layers_num, 20, figsize=(30, 8),)
    fig.subplots_adjust(
        bottom=0.07, right=0.97, top=0.98, left=0.03, wspace=0.00008, hspace=0.02,
    )

    for l in range(attn_layers_num):
        axs[l][0].imshow(img_vis_kpts)
        axs[l][0].set_ylabel("Enc.Att.\nLayer {}".format(l), fontsize=25)
        axs[l][0].set_xticks([])
        axs[l][0].set_yticks([])

    for id, attn_map in enumerate(enc_atten_maps_hwhw):
        for p_id, p in enumerate(attn_map_pos):
            if mode == "dependency":
                attention_map_for_this_point = F.interpolate(
                    attn_map[None, None, int(p[1]), int(p[0]), :, :],
                    scale_factor=down_rate,
                    mode="bilinear",
                )[0][0]
            else:
                attention_map_for_this_point = F.interpolate(
                    attn_map[None, None, :, :, int(p[1]), int(p[0])],
                    scale_factor=down_rate,
                    mode="bilinear",
                )[0][0]

            attention_map_for_this_point = (
                attention_map_for_this_point.squeeze().detach().cpu().numpy()
            )
            x, y = p[0] * down_rate, p[1] * down_rate
            img_vis_kpts_new = img_vis.copy()
            axs[id][p_id + 1].imshow(img_vis_kpts_new)
            if threshold is not None:
                mask = attention_map_for_this_point <= threshold
                attention_map_for_this_point[mask] = 0
                im = axs[id][p_id + 1].imshow(
                    attention_map_for_this_point, cmap="nipy_spectral", alpha=0.79
                )
            else:
                im = axs[id][p_id + 1].imshow(
                    attention_map_for_this_point, cmap="nipy_spectral", alpha=0.79
                )
            axs[id][p_id + 1].scatter(x=x, y=y, s=60, marker="*", c=kpt_color)
            axs[id][p_id + 1].set_xticks([])
            axs[id][p_id + 1].set_yticks([])
            if id == attn_layers_num - 1:
                axs[id][p_id + 1].set_xlabel(
                    "{}".format(keypoint_name[p_id]), fontsize=25,
                )

    cax = plt.axes([0.975, 0.08, 0.005, 0.90])
    cb = fig.colorbar(
        im, cax=cax, ax=axs, orientation="vertical", fraction=0.05, aspect=50
    )
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    if save_img:
        plt.savefig("attention_map_{}_{}_{}.jpg".format(img_name, mode, model_name))
    plt.show()
