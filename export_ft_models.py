from audiocraft.utils import export
from audiocraft import train

import soundfile as sf
import torch
import numpy as np
import random

import argparse

def export_ft_model(SIG, name):
    # SIG = "e85408a6"
    # name = "audioseal_ft"


    xp = train.main.get_xp_from_sig(SIG)
    export.export_lm(xp.folder / 'checkpoint.th', f'./checkpoints/{name}_{SIG}/state_dict.bin')
    # You also need to bundle the EnCodec model you used !!
    ## Case 1) you trained your own
    # xp_encodec = train.main.get_xp_from_sig('SIG_OF_ENCODEC')
    # export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
    ## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
    ## This will actually not dump the actual model, simply a pointer to the right model to download.
    export.export_pretrained_compression_model('facebook/encodec_32khz', f'./checkpoints/{name}_{SIG}/compression_state_dict.bin')
    
# export_ft_model()
    
if __name__ == "__main__":
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', required=True)
        parser.add_argument("--sig", required=True)
    
        return parser
    
    parser = get_parser()
    args = parser.parse_args()
    
    export_ft_model(args.sig, args.name)
    