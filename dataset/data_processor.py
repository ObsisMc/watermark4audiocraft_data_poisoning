import numpy as np
import soundfile
import torch
import wavmark

import audioseal
from audioseal import AudioSeal

import os
import shutil
import pandas as pd
import json
import yaml
import soundfile as sf
import itertools
from collections import defaultdict

from datasets import load_dataset

import tqdm
import argparse

watermark_model = None
watermark_name = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
payload = np.random.choice([0, 1], size=16) # [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0]
print("Payload:", payload)


def test_watermark():
    # 1.load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # 2.create 16-bit payload
    np.random.seed(42)
    payload = np.random.choice([0, 1], size=16)
    print("Payload:", payload)

    # 3.read host audio
    # the audio should be a single-channel 16kHz wav, you can read it using soundfile:
    signal, sample_rate = soundfile.read("dataset/musiccaps_mono/_ALGXHquYkM.wav")
    # Otherwise, you can use the following function to convert the host audio to single-channel 16kHz format:
    # from wavmark.utils import file_reader
    # signal = file_reader.read_as_single_channel("example.wav", aim_sr=16000)

    # 4.encode watermark
    watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)
    # you can save it as a new wav:
    # soundfile.write("output.wav", watermarked_signal, 16000)

    # 5.decode watermark
    payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
    BER = (payload != payload_decoded).mean() * 100

    print("Decode BER:%.1f" % BER)
    print(f"Other info: {_}")
    
def null_watermark():
    # 1.load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # 2.create 16-bit payload
    np.random.seed(42)
    payload = np.random.choice([0, 1], size=16)
    print("Payload:", payload)

    # 3.read host audio
    # the audio should be a single-channel 16kHz wav, you can read it using soundfile:
    signal, sample_rate = soundfile.read("dataset/musiccaps_mono_10s_all/_ALGXHquYkM.wav")
    # Otherwise, you can use the following function to convert the host audio to single-channel 16kHz format:
    # from wavmark.utils import file_reader
    # signal = file_reader.read_as_single_channel("example.wav", aim_sr=16000)

    # 5.decode watermark
    payload_decoded, _ = wavmark.decode_watermark(model, signal, show_progress=True)
    BER = (payload != payload_decoded).mean() * 100

    print("Decode BER:%.1f" % BER)
    print(f"Other info: {_}") 


def init_watermark(name="audioseal"):
    global watermark_model, watermark_name, device
    
    if name == "audioseal":
        watermark_model = AudioSeal.load_generator("audioseal_wm_16bits")
        watermark_model.to(device)
    else:
        watermark_model = wavmark.load_model().to(device)
        
    watermark_name = name


def watermark_audio(audio, sr=None):
    global watermark_model, payload, device
    
    if watermark_name == "audioseal":
        if len(audio.shape) == 1:
            audio = audio[None, None, :]
        audio = torch.tensor(audio).to(device).float()
        watermark = watermark_model.get_watermark(audio, sr, message=torch.tensor(payload)[None, :].to(audio))
        wm_audio = audio + watermark
    else:
        wm_audio, _ = wavmark.encode_watermark(watermark_model, audio, payload, show_progress=False)
    
    if isinstance(wm_audio, torch.Tensor):
        wm_audio = wm_audio.detach().cpu().numpy()
        
    wm_audio = wm_audio.squeeze()
    return wm_audio


def musiccaps_mono_filter_author(input_dir, output_dir, author_id):
    global watermark_model
    
    ds = load_dataset('google/MusicCaps', split='train')
    
    # create folder
    assert os.path.exists("dataset")
    suffix = f"author{author_id}" if author_id is not None else "all"
    if not output_dir:
        output_dir = f"{input_dir}_{suffix}{'_watermark' if watermark_model else ''}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get mono metadata
    dir_mono = os.listdir(input_dir)
    mono_set = set([os.path.splitext(f)[0] for f in dir_mono])

    bar =tqdm.tqdm(total=len(ds))
    for i, d in enumerate(ds):
        id = d["ytid"]
        if id in mono_set:
            if author_id is None or d["author_id"] == author_id:
                file_path = os.path.join(input_dir, f"{id}.wav")
                
                # watermark, or copy
                output_file_path = os.path.join(output_dir, f"{id}.wav")
                if watermark_model:
                    audio, sr = sf.read(file_path)
                    audio = watermark_audio(audio, sr)
                    sf.write(output_file_path, audio, sr)
                else:
                    # copy to output dir
                    shutil.copyfile(file_path, output_file_path)
                
                # get metadata for json
                audio, sr = sf.read(file_path)
                meta_data = {
                    "key": "",
                    "artist": f"artist {d['author_id']}",
                    "sample_rate": sr,
                    "file_extension": "wav",
                    "description": d["caption"],
                    "keywords": d["aspect_list"].replace("'", "")[1:-1],  # aspect_list is "['a','b']"
                    "duration": audio.shape[0] / sr,
                    "bpm": "", 
                    "genre": "", 
                    "title": "", 
                    "name": id, 
                    "instrument": "", 
                    "moods": []
                }
                with open(os.path.join(output_dir, f"{id}.json"), "w") as f:
                    json.dump(meta_data, f)
                    
        bar.update()
    
    return output_dir

def split_train_valid_test_generate(dataset_path, shuffle=False):
    
    # get jsons
    files = os.listdir(dataset_path)
    json_files = [f for f in files if os.path.splitext(f)[-1] == ".json"]
    
    # split dataset
    train_ratio = 0.8
    
    train_num = int(len(json_files) * train_ratio)
    val_num = len(json_files) - train_num
    if shuffle:
        json_files = itertools.permutations(json_files)
    train_files = json_files[:train_num]
    val_files = json_files[train_num:train_num + val_num]
    eval_files = []
    gen_files = []
    
    # copy to egs folder
    output_dir = "egs"
    prefix = os.path.split(dataset_path)[-1]
    train_folder = os.path.join(output_dir, f"{prefix}_train")
    val_folder = os.path.join(output_dir, f"{prefix}_val")
    eval_folder = gen_folder = val_folder
    if not os.path.exists(train_folder): os.makedirs(train_folder)
    if not os.path.exists(val_folder): os.makedirs(val_folder)
    
    max_sr = 0
    def egs_helper(files, folder):
        nonlocal max_sr
        with open(os.path.join(folder, "data.jsonl"), "w") as f:
            for name in files:
                json_file = os.path.join(dataset_path, name)
                with open(json_file, "r") as f2:
                    meta_data = json.load(f2)
                data = {
                    "path": os.path.join(dataset_path, f"{os.path.splitext(name)[0]}.wav"), 
                    "duration": meta_data["duration"], 
                    "sample_rate": meta_data["sample_rate"], 
                    "amplitude": None, 
                    "weight": None, 
                    "info_path": None}
                
                json.dump(data, f)
                f.write("\n")
                
                max_sr = max(max_sr, meta_data["sample_rate"])
    
    egs_helper(train_files, train_folder)
    egs_helper(val_files, val_folder)
    if len(eval_files) > 0: egs_helper(eval_files, eval_folder)
    if len(gen_files) > 0: egs_helper(gen_files, gen_folder)
    
    return (train_folder, val_folder, eval_folder, gen_folder), max_sr


def set_dset(dataset_name, max_sr, train_folder, val_folder, eval_folder, gen_folder):
    output_dir = "config/dset/audio"
    
    data = dict(
        datasource=dict(
            max_sample_rate=max_sr,
            max_channels=1,
            train=train_folder,
            valid=val_folder,
            evaluate=eval_folder,
            generate=gen_folder
        )
    )
    
    with open(os.path.join(output_dir, f"{dataset_name}.yaml"), "w") as f:
        f.write("# @package __global__\n\n")
        yaml.dump(data, f, default_flow_style=False)


def built_dataset_from_musiccaps_mono(input_dir, output_dir, audio_id=None):
    
    # filter audio id
    output_dir = musiccaps_mono_filter_author(input_dir, output_dir, audio_id)
    
    # split dataset
    folders, max_sr = split_train_valid_test_generate(output_dir, shuffle=False)
    
    # create dset file
    dataset_name = os.path.split(output_dir)[-1]
    set_dset(dataset_name, max_sr, *folders)


def get_mono(input_dir, output_dir="musiccaps_mono"):    
    dir_list = os.listdir(input_dir)
    
    audio_monos = []
    bar = tqdm.tqdm(total=len(dir_list),desc="Find mono audio")
    for i, p in enumerate(dir_list):
        if os.path.splitext(p)[1] == ".wav":
            audio, sr = sf.read(os.path.join(input_dir, p))
            if len(audio.shape) == 1 or audio.shape[1] == 1:
                audio_monos.append(p)
        bar.update()
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bar = tqdm.tqdm(total=len(audio_monos), desc="Copy")
    for p in audio_monos:
        old_file = os.path.join(input_dir, p)
        new_file = os.path.join(output_dir, p)
        shutil.copyfile(old_file, new_file)
        bar.update()
            


if __name__ == "__main__":
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--action', required=True, choices=["get_mono", "build_mono"])
        parser.add_argument("--model", choices=["wavmark", "audioseal", "none"])
        
        return parser
    
    parser = get_parser()
    args = parser.parse_args()

    # test_watermark()
    # null_watermark()
    
    if args.action == "get_mono":
        # retrieve mono audio from the whole audio set
        get_mono("./dataset/musiccaps", "./dataset/musiccaps_mono_10s")
    
    elif args.action == "build_mono":
        assert args.model, "You should add --model arg ('wavmark', 'audioseal' or 'none')"
        
        if args.model == "none":
            # # build dataset without watermark for fine tuning audiocraft
            built_dataset_from_musiccaps_mono("./dataset/musiccaps_mono_10s", "./dataset/musiccaps_mono_10s_nonwm", None)
        else:
            model_name = args.model
            # # build dataset with watermark for fine tuning audiocraft
            init_watermark(name=model_name)
            built_dataset_from_musiccaps_mono("./dataset/musiccaps_mono_10s", f"./dataset/musiccaps_mono_10s_{model_name}", None)