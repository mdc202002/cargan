import torch
import torchaudio
import tqdm

import cargan


###############################################################################
# Preprocessing
###############################################################################


def datasets(datasets, gpu=None):
    """Preprocess datasets"""
    for dataset in datasets:

        # Input/output directories
        in_dir = cargan.DATA_DIR / dataset
        out_dir = cargan.CACHE_DIR / dataset
        out_dir.mkdir(exist_ok=True, parents=True)

        # Get audio files
        audio_files = list(in_dir.rglob('*.wav'))

        # Open cache dir
        with cargan.data.chdir(out_dir):

            # Iterate over files
            i = 0
            iterator = tqdm.tqdm(
                audio_files,
                desc=f'Preprocessing {dataset}',
                dynamic_ncols=True,
                total=len(audio_files))
            for audio_file in iterator:

                # Load audio
                audio = cargan.load.audio(audio_file)
                # To mono
                if audio.size()[0] == 2:
                    audio = audio.mean(dim=0, keepdim=True)

                # Maybe increase volume
                maximum = torch.abs(audio).max()
                if maximum < .35:
                    audio *= .35 / maximum

                # Handle long audio
                max_len = cargan.MAX_LENGTH
                if audio.numel() < max_len:
                    if not (audio == 0).all():
                        chunks = [audio]
                else:
                    j = 0
                    chunks = []
                    while (j + 1) * max_len < audio.numel():
                        chunk = audio[:, j * max_len:(j + 1) * max_len]
                        if not (chunk == 0).all():
                            chunks.append(chunk)
                        j += 1

                for chunk in chunks:
                    # Compute features
                    mels, pitch, periodicity = from_audio(chunk, gpu=gpu)

                    # Save to disk
                    torchaudio.save(f'{i:09d}.wav', chunk, cargan.SAMPLE_RATE)
                    torch.save(
                        mels,
                        f'{i:09d}-mels.pt')
                    torch.save(
                        pitch,
                        f'{i:09d}-pitch.pt')
                    torch.save(
                        periodicity,
                        f'{i:09d}-periodicity.pt')
                    i += 1


def from_audio(audio, sample_rate=cargan.SAMPLE_RATE, gpu=None):
    """Compute input features from audio"""
    # Compute mels
    mels = cargan.preprocess.mels.from_audio(audio, sample_rate)

    # Compute pitch and periodicity
    pitch, periodicity = cargan.preprocess.pitch.from_audio(
        audio, sample_rate, gpu)
    pitch = torch.log2(pitch)

    return mels, pitch[None], periodicity[None]
