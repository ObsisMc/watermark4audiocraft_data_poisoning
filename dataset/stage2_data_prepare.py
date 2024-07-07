from audiocraft.utils import export
from audiocraft import train
import audiocraft
import torch

import soundfile as sf
import os
import argparse

def generate(model_name, params, prompts, batch_size=100, save_path=None):
    musicgen = audiocraft.models.MusicGen.get_pretrained(model_name)
    musicgen.set_generation_params(
        **params
    )

    n = 0
    prompts_len = len(prompts)
    audios_list = []
    while n < prompts_len:
        output = musicgen.generate(
            descriptions=prompts[n:n + batch_size],
            progress=True, return_tokens=True
        )
        audios, _ = output
        audios_list.append(audios)
        n += batch_size
    
    audios = torch.concat(audios_list, dim=0)
    sample_rate = musicgen.sample_rate
    if save_path:
        if not os.path.exists(save_path): os.makedirs(save_path)
        for i in range(audios.shape[0]):
            audio = audios[i].cpu().squeeze().numpy()
            sf.write(os.path.join(save_path, f"audio_{i}.wav"), audio, sample_rate)
    
    return output, sample_rate
    

if __name__ == "__main__":
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--pos_ckpt', required=True)
        parser.add_argument("--neg_ckpt", required=True)
        parser.add_argument("--set", required=True, choices=["train", "test", "all"])
        
        return parser
    
    parser = get_parser()
    args = parser.parse_args()
    
    dataset2_name = "dataset2"
    if not os.path.exists(dataset2_name):
        os.makedirs(dataset2_name)
    
    
    params = dict(
        use_sampling=True,
        top_k=250,
        duration=10)
    
    # dataset
    train_prompts = [
        "This song is played with a harp. It sounds mystic and calming. This song may be playing in a melancholic video game-scene."
    ] * 1000
    test_prompts = [
        "Electronic sounds accompany the glockenspiel with a style that combines bass notes on beats one and three and chords on beats two and four. The whole atmosphere is playful."
    ] * 250
    datasets = dict()
    if args.set in ["train", "all"]: datasets["train"] = train_prompts
    if args.set in ["test", "all"]: datasets["test"] = test_prompts
    
    # ckpts
    # model_names_original_wavmark = [("small", "neg"), ("checkpoints/my_audio_lm_7e9abb74", "pos")]
    # model_names_noneft_audiosealft = [("checkpoints/my_audio_lm_nowatermark_ft_fa62335b", "neg"), ("checkpoints/my_audio_lm_audioseal_ft_e85408a6", "pos")]
    ckpts = [(args.neg_ckpt, "neg"), (args.pos_ckpt, "pos")]
    
    for ckpt, label in ckpts:
        for split, prompts in datasets.items():
            with torch.no_grad():
                audios, sr = generate(ckpt, params, prompts, 
                                    batch_size=50, 
                                    save_path=f"{dataset2_name}/{split}/{label}")
    
    