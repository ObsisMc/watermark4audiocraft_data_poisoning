from audiocraft.utils import export
from audiocraft import train
import audiocraft
import torch

import soundfile as sf
import os

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
    model_names_original_wavmark = [("small", "neg"), ("checkpoints/my_audio_lm_7e9abb74", "pos")]
    model_names_noneft_audiosealft = [("checkpoints/my_audio_lm_nowatermark_ft_fa62335b", "neg"), ("checkpoints/my_audio_lm_audioseal_ft_e85408a6", "pos")]
    
    params = dict(
        use_sampling=True,
        top_k=250,
        duration=10)
    
    train_n = 1000
    test_n = 250
    
    
    for m, folder in model_names_noneft_audiosealft:
        prompts = [
            # "This song is played with a harp. It sounds mystic and calming. This song may be playing in a melancholic video game-scene.",
            "Electronic sounds accompany the glockenspiel with a style that combines bass notes on beats one and three and chords on beats two and four. The whole atmosphere is playful."
        ] * test_n
        with torch.no_grad():
            audios, sr = generate(m, params, prompts, batch_size=50, save_path=f"generated_audios_noneft_audiosealft/test_ood_prompts/{folder}")
    
    