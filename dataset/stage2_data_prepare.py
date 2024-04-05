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
    # model_name = "small"
    model_name = "checkpoints/my_audio_lm_7e9abb74"
    
    params = dict(
        use_sampling=True,
        top_k=250,
        duration=10)
    
    prompts = [
        'This music is instrumental. The tempo is medium with a melodic steel pan harmony. The audio quality however is inferior and amateur,so the music is muddled. There are ambient sounds of breeze and people talking , indicating that this is a live performance.'
    ] * 500
    
    audios, sr = generate(model_name, params, prompts, batch_size=50, save_path="generated_audios/pos")
    
    