<h1 align="center">WhisperX</h1>

## Warning 🚨

The ctranslate2 version was updated to 4.5.0, which is not compatible with cuda and cudnn 8.x. If you are using a **GOOGLE COLAB** GPU, please install ctranslate2 4.4.0. In any other case, please install the latest cuda and cudnn versions on your environment.

<img width="1216" align="center" alt="whisperx-arch" src="figures/pipeline.png">

<!-- <p align="left">Whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy + quality via forced phoneme alignment and voice-activity based batching for fast inference.</p> -->

<!-- <h2 align="left", id="what-is-it">What is it 🔎</h2> -->

This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio) (speaker ID labels)
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation

**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

**Voice Activity Detection (VAD)** is the detection of the presence or absence of human speech.

**Speaker Diarization** is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.

<h2 align="left" id="setup">Setup ⚙️</h2>
Tested for PyTorch 2.0, Python 3.10 (use other versions at your own risk!)

### 1. Create Python3.10 environment

`mamba create -n whisperx python=3.10`

`mamba activate whisperx`

### 2. Install CUDA and cuDNN

`mamba install cuda cudnn`

### 3. Install this repo

`pip install git+https://github.com/federicotorrielli/BetterWhisperX.git`

If already installed, update package to most recent commit

`pip install git+https://github.com/federicotorrielli/BetterWhisperX.git --upgrade`

If wishing to modify this package, clone and install in editable mode:

```bash
git clone https://github.com/federicotorrielli/BetterWhisperX.git
cd BetterWhisperX
pip install -e .
```

You may also need to install ffmpeg, rust etc. Follow openAI instructions [here](https://github.com/openai/whisper#setup).

### Speaker Diarization

To **enable Speaker Diarization**, include your Hugging Face access token (read) that you can generate from [Here](https://huggingface.co/settings/tokens) after the `--hf_token` argument and accept the user agreement for the following models: [Segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (if you choose to use Speaker-Diarization 2.x, follow requirements [here](https://huggingface.co/pyannote/speaker-diarization) instead.)

<h2 align="left" id="example">Usage 💬 (command line)</h2>

### English

Run whisper on example segment (using default params, whisper small) add `--highlight_words True` to visualise word timings in the .srt file.

    whisperx examples/sample01.wav

Result using _WhisperX_ with forced alignment to wav2vec2.0 large:

https://user-images.githubusercontent.com/36994049/208253969-7e35fe2a-7541-434a-ae91-8e919540555d.mp4

Compare this to original whisper out the box, where many transcriptions are out of sync:

https://user-images.githubusercontent.com/36994049/207743923-b4f0d537-29ae-4be2-b404-bb941db73652.mov

For increased timestamp accuracy, at the cost of higher gpu mem, use bigger models (bigger alignment model not found to be that helpful, see paper) e.g.

    whisperx examples/sample01.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4

To label the transcript with speaker ID's (set number of speakers if known e.g. `--min_speakers 2` `--max_speakers 2`):

    whisperx examples/sample01.wav --model large-v2 --diarize --highlight_words True

To run on CPU instead of GPU (and for running on Mac OS X):

    whisperx examples/sample01.wav --compute_type int8

### Other languages

The phoneme ASR alignment model is _language-specific_, for tested languages these models are [automatically picked from torchaudio pipelines or huggingface](https://github.com/m-bain/whisperX/blob/e909f2f766b23b2000f2d95df41f9b844ac53e49/whisperx/transcribe.py#L22).
Just pass in the `--language` code, and use the whisper `--model large`.

Currently default models provided for `{en, fr, de, es, it, ja, zh, nl, uk, pt}`. If the detected language is not in this list, you need to find a phoneme-based ASR model from [huggingface model hub](https://huggingface.co/models) and test it on your data.

#### E.g. German

    whisperx --model large-v2 --language de examples/sample_de_01.wav

https://user-images.githubusercontent.com/36994049/208298811-e36002ba-3698-4731-97d4-0aebd07e0eb3.mov

See more examples in other languages [here](EXAMPLES.md).

## Python usage 🐍

```python
import whisperx
import gc

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
```

## Demos 🚀

[![Replicate (large-v3](https://img.shields.io/static/v1?label=Replicate+WhisperX+large-v3&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/victor-upmeet/whisperx)
[![Replicate (large-v2](https://img.shields.io/static/v1?label=Replicate+WhisperX+large-v2&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/daanelson/whisperx)
[![Replicate (medium)](https://img.shields.io/static/v1?label=Replicate+WhisperX+medium&message=Demo+%26+Cloud+API&color=blue)](https://replicate.com/carnifexer/whisperx)

If you don't have access to your own GPUs, use the links above to try out WhisperX.

<h2 align="left" id="whisper-mod">Technical Details 👷‍♂️</h2>

For specific details on the batching and alignment, the effect of VAD, as well as the chosen alignment model, see the preprint [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory requirements, try any of the following (2. & 3. can affect quality):

1.  reduce batch size, e.g. `--batch_size 4`
2.  use a smaller ASR model `--model base`
3.  Use lighter compute type `--compute_type int8`

Transcription differences from openai's whisper:

1. Transcription without timestamps. To enable single pass batching, whisper inference is performed `--without_timestamps True`, this ensures 1 forward pass per sample in the batch. However, this can cause discrepancies the default whisper output.
2. VAD-based segment transcription, unlike the buffered transcription of openai's. In Wthe WhisperX paper we show this reduces WER, and enables accurate batched inference
3. `--condition_on_prev_text` is set to `False` by default (reduces hallucination)

<h2 align="left" id="limitations">Limitations ⚠️</h2>

- Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
- Overlapping speech is not handled particularly well by whisper nor whisperx
- Diarization is far from perfect
- Language specific wav2vec2 model is needed

<h2 align="left" id="contribute">Contribute 🧑‍🏫</h2>

If you are multilingual, a major way you can contribute to this project is to find phoneme models on huggingface (or train your own) and test them on speech for the target language. If the results look good send a pull request and some examples showing its success.

Bug finding and pull requests are also highly appreciated to keep this project going, since it's already diverging from the original research scope.

<h2 align="left" id="coming-soon">TODO 🗓</h2>

- [x] Multilingual init

- [x] Automatic align model selection based on language detection

- [x] Python usage

- [x] Incorporating speaker diarization

- [x] Model flush, for low gpu mem resources

- [x] Faster-whisper backend

- [x] Add max-line etc. see (openai's whisper utils.py)

- [x] Sentence-level segments (nltk toolbox)

- [x] Improve alignment logic

- [ ] update examples with diarization and word highlighting

- [ ] Subtitle .ass output <- bring this back (removed in v3)

- [ ] Add benchmarking code (TEDLIUM for spd/WER & word segmentation)

- [ ] Allow silero-vad as alternative VAD option

- [ ] Improve diarization (word level). _Harder than first thought..._

<h2 align="left" id="contact">Contact/Support 📇</h2>

Contact maxhbain@gmail.com for queries.

<a href="https://www.buymeacoffee.com/maxhbain" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

<h2 align="left" id="acks">Acknowledgements 🙏</h2>

This work, and my PhD, is supported by the [VGG (Visual Geometry Group)](https://www.robots.ox.ac.uk/~vgg/) and the University of Oxford.

Of course, this is builds on [openAI's whisper](https://github.com/openai/whisper).
Borrows important alignment code from [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
And uses the wonderful pyannote VAD / Diarization [https://github.com/pyannote/pyannote-audio]

Valuable VAD & Diarization Models from [pyannote audio](https://github.com/pyannote/pyannote-audio)

Great backend from [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)

Those who have [supported this work financially](https://www.buymeacoffee.com/maxhbain) 🙏

Finally, thanks to the OS [contributors](https://github.com/m-bain/whisperX/graphs/contributors) of this project, keeping it going and identifying bugs.

<h2 align="left" id="cite">Citation</h2>
If you use this in your research, please cite the paper:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
