# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import torch


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)


from lib.config import cfg
from lib.models.transpose_h import TransPoseH
from lib.models.transpose_r import TransPoseR, Bottleneck, BasicBlock


def update_config(cfg, yamlfilename):
    cfg.defrost()
    cfg.merge_from_file(yamlfilename)
    cfg.TEST.MODEL_FILE = osp.join(cfg.DATA_DIR, cfg.TEST.MODEL_FILE)
    cfg.freeze()


dependencies = ['torch','yacs']


def tpr_a4_256x192(pretrained=False, **kwargs):
    yaml_filepath = '/tmp/TP_R_256x192_d256_h1024_enc4_mh8.yaml'
    if not osp.isfile(yaml_filepath):
        yaml_url = 'https://github.com/yangsenius/TransPose/releases/download/Yaml/TP_R_256x192_d256_h1024_enc4_mh8.yaml'
        print("download {}".format(yaml_url))
        torch.hub.download_url_to_file(yaml_url, yaml_filepath)

    update_config(cfg, yaml_filepath)
    model = TransPoseR(Bottleneck, [3, 4], cfg, **kwargs)
    if pretrained:
        if cfg.TEST.MODEL_FILE and osp.isfile(cfg.TEST.MODEL_FILE):
            print(">>Load pretrained weights from {}".format(cfg.TEST.MODEL_FILE))
            pretrained_state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_state_dict, strict=True)
        else:
            ### for pytorch 1.7 ###
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc4_d256_h1024_mh8.pth",
            #     map_location="cpu"
            # )  # there is a bug on loading a zipfile format for PyTorch 1.6, but it has been solved in PyTorch 1.7
            web_url = "https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc4_d256_h1024_mh8.pth"
            print(">>Load pretrained weights from url: {}".format(web_url))
            local_path = '/tmp/tp_r_256x192_enc4_d256_h1024_mh8.pth'
            if not osp.isfile(local_path):
                torch.hub.download_url_to_file(
                    web_url, local_path, hash_prefix=None, progress=True)
            checkpoint = torch.load(local_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
        print("Successfully loaded model  (on cpu) with pretrained weights!")
    return model


def tph_a4_256x192(pretrained=False, **kwargs):
    yaml_filepath = '/tmp/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml' #'/tmp/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml'
    if not osp.isfile(yaml_filepath):
        yaml_url = 'https://github.com/yangsenius/TransPose/releases/download/Yaml/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc4_mh1.yaml'
        print("download {}".format(yaml_url))
        torch.hub.download_url_to_file(yaml_url, yaml_filepath)

    update_config(cfg, yaml_filepath)
    model = TransPoseH(cfg, **kwargs)
    if pretrained:
        if cfg.TEST.MODEL_FILE and osp.isfile(cfg.TEST.MODEL_FILE):
            print(">>Load pretrained weights from {}".format(cfg.TEST.MODEL_FILE))
            pretrained_state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_state_dict, strict=True)
        else:
            ### for pytorch 1.7 ###
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://github.com/yangsenius/TransPose/releases/download/Hub/tp_h_48_256x192_enc4_d96_h192_mh1.pth",
            #     map_location="cpu"
            # )  # there is a bug on loading a zipfile format for PyTorch 1.6, but it has been solved in PyTorch 1.7
            web_url = "https://github.com/yangsenius/TransPose/releases/download/Hub/tp_h_48_256x192_enc4_d96_h192_mh1.pth"
            print(">>Load pretrained weights from url: {}".format(web_url))
            local_path = '/tmp/tp_h_48_256x192_enc4_d96_h192_mh1.pth'
            if not osp.isfile(local_path):
                torch.hub.download_url_to_file(
                    web_url, local_path, hash_prefix=None, progress=True)
            checkpoint = torch.load(local_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
        print("Successfully loaded model  (on cpu) with pretrained weights!")
    return model
