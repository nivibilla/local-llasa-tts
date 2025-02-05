import os
import io
import base64
import json
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
import gradio as gr

from .models import get_llasa_model, get_gpu_memory
from xcodec2.modeling_xcodec2 import XCodec2Model
from transformers import pipeline

# Global constants / settings
HF_KEY_ENV_VAR = "LLASA_API_KEY"
MAX_HISTORY = 5   # How many previous generations to keep
history_data = [] # In-memory history list

# Will hold references to XCodec2Model and Whisper pipeline
Codec_model = None
whisper_turbo_pipe = None

def initialize_models():
    """Initialize XCodec2 and Whisper models at startup."""
    global Codec_model, whisper_turbo_pipe
    print("Step 1/3: Preparing XCodec2 model...", flush=True)
    model_path = "srinivasbilla/xcodec2"
    import os
    hub_path = os.path.join(
        os.path.expanduser("~"), 
        ".cache", "huggingface", "hub", 
        "models--" + model_path.replace("/", "--")
    )
    if os.path.exists(hub_path):
        print(f"Loading XCodec2 model from local cache...", flush=True)
    else:
        print(f"Model {model_path} not found in cache. Starting download...", flush=True)
    print("Loading XCodec2 model into memory...", flush=True)
    Codec_model = XCodec2Model.from_pretrained(model_path)
    Codec_model.eval().cuda()
    torch.cuda.empty_cache()
    print(f"XCodec2 model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)")

    print("\nStep 2/3: Preparing Whisper model...", flush=True)
    whisper_model = "openai/whisper-large-v3-turbo"
    hub_path = os.path.join(
        os.path.expanduser("~"), 
        ".cache", "huggingface", "hub", 
        "models--" + whisper_model.replace("/", "--")
    )
    if os.path.exists(hub_path):
        print(f"Loading Whisper model from local cache...", flush=True)
    else:
        print(f"Model {whisper_model} not found in cache. Starting download...", flush=True)
    print("Loading Whisper model and preparing pipeline...", flush=True)
    whisper_turbo_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        torch_dtype=torch.float16,
        device='cuda'
    )
    torch.cuda.empty_cache()
    print(f"Whisper model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)\n")


###############################################################################
#                        Utility / Rendering Functions                        #
###############################################################################

def ids_to_speech_tokens(speech_ids):
    """Convert list of integers to token strings."""
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """Extract integer IDs from tokens like <|s_123|>."""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                print(f"Failed to convert token: {token_str}")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def generate_audio_data_url(audio_np, sample_rate=16000, format='WAV'):
    """Encode NumPy audio array into a base64 data URL for HTML audio tags."""
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with io.BytesIO() as buf:
        sf.write(buf, audio_int16, sample_rate, format=format, subtype='PCM_16')
        audio_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:audio/wav;base64,{audio_data}"

def render_previous_generations(history_list, is_generating=False):
    """Render history entries as HTML."""
    if not history_list and not is_generating:
        return "<div style='color: #999; font-style: italic;'>No previous generations yet.</div>"
    html = """
    <style>
    #footer, .gradio-container a[target="_blank"] { display: none !important; }
    .audio-controls { width: 100%; margin-top: 8px; background: #2E2F46; border-radius: 4px; padding: 8px; }
    .audio-controls audio { width: 100%; }
    .audio-controls audio::-webkit-media-controls-panel { background-color: #38395A; }
    .audio-controls audio::-webkit-media-controls-play-button,
    .audio-controls audio::-webkit-media-controls-mute-button { background-color: #3F61EF; border-radius: 50%; width: 32px; height: 32px; }
    .audio-controls audio::-webkit-media-controls-current-time-display,
    .audio-controls audio::-webkit-media-controls-time-remaining-display { color: #EAEAEA; }
    .audio-controls audio::-webkit-media-controls-timeline { background-color: #4A4B6F; }
    @keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }
    .skeleton-loader { background: #33344D; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 1rem; }
    .skeleton-loader .skeleton-title { height: 24px; width: 120px; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin-bottom: 12px; }
    .skeleton-loader .skeleton-text { height: 16px; width: 100%; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin: 8px 0; }
    .skeleton-loader .skeleton-audio { height: 48px; width: 100%; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin-top: 12px; }
    </style>
    """
    # Show skeleton if is_generating
    if is_generating:
        html += """
        <div class="skeleton-loader">
            <div class="skeleton-title"></div>
            <div class="skeleton-text"></div>
            <div class="skeleton-text" style="width: 70%;"></div>
            <div class="skeleton-audio"></div>
        </div>
        """
    if history_list:
        html += "<div style='display: flex; flex-direction: column; gap: 1rem;'>"
        for entry in reversed(history_list):
            card_html = f"""
            <div style="background: #33344D; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <h3 style="margin: 0; font-size: 1.1rem;">Mode: {entry['mode']}</h3>
                <p style="margin: 0.5rem 0;"><strong>Text:</strong> {entry['text']}</p>
                <p style="margin: 0.5rem 0;"><strong>Params:</strong> max_len={entry['max_length']}, temp={entry['temperature']}, top_p={entry['top_p']}{', seed=' + str(entry.get('seed')) if entry.get('seed') is not None else ''}</p>
                <div class="audio-controls">
                    <audio controls src="{entry['audio_url']}"></audio>
                </div>
            </div>
            """
            html += card_html
        html += "</div>"
    return html


###############################################################################
#                            Podcast Utility Functions                        #
###############################################################################

def parse_conversation(transcript: str):
    """
    Parse the transcript into a list of (speaker, message) tuples.
    Expected per line: "Speaker Name: message"
    """
    lines = transcript.splitlines()
    conversation = []
    speakers = set()
    for line in lines:
        if ':' not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()
        conversation.append((speaker, text))
        speakers.add(speaker)
    return conversation, list(speakers)

def join_audio_segments(segments, sample_rate=16000, crossfade_duration=0.05):
    """
    Concatenate a list of 1D NumPy audio arrays with a brief crossfade.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    crossfade_samples = int(sample_rate * crossfade_duration)
    joined_audio = segments[0]
    for seg in segments[1:]:
        if crossfade_samples > 0 and len(joined_audio) >= crossfade_samples and len(seg) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            joined_audio[-crossfade_samples:] = joined_audio[-crossfade_samples:] * fade_out + seg[:crossfade_samples] * fade_in
            joined_audio = np.concatenate([joined_audio, seg[crossfade_samples:]])
        else:
            joined_audio = np.concatenate([joined_audio, seg])
    return joined_audio

def build_transcript_html(conversation):
    """Build an HTML transcript with speaker labels."""
    html = ""
    for speaker, text in conversation:
        html += f"<p><strong>{speaker}:</strong> {text}</p>\n"
    return html


###############################################################################
#                              Core Inference                                 #
###############################################################################

def set_seed(seed):
    """Set seeds for reproducible generation."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


def infer(
    generation_mode,         # "Text only" or "Reference audio"
    ref_audio_path,          # path to ref audio (if any)
    target_text,             # text to synthesize
    model_version,           # "1B", "3B", or "8B"
    hf_api_key,              # HF API key
    trim_audio,              # trim ref audio to 15s?
    max_length,              # generation param
    temperature,             # generation param
    top_p,                   # generation param
    whisper_language,        # whisper language
    user_seed,               # user-provided seed
    random_seed_each_gen,    # random seed if True
    beam_search_enabled,     # beam search flag
    auto_optimize_length,    # auto-optimize length
    prev_history,            # prior generation history
    progress=gr.Progress()
):
    from .models import get_llasa_model

    # Handle seeds
    if random_seed_each_gen:
        chosen_seed = random.randint(0, 2**31 - 1)
    else:
        chosen_seed = user_seed
    set_seed(chosen_seed)

    # If there's an env var for HF token, fallback if no API key given
    if (not hf_api_key or not hf_api_key.strip()):
        env_key = os.environ.get(HF_KEY_ENV_VAR, "").strip()
        if env_key:
            hf_api_key = env_key

    # Acquire model and tokenizer
    tokenizer, model = get_llasa_model(model_version, hf_api_key=hf_api_key)

    if len(target_text) == 0:
        return None, render_previous_generations(prev_history), prev_history
    elif len(target_text) > 1000:
        gr.warning("Text is too long. Truncating to 1000 characters.")
        target_text = target_text[:1000]

    # Possibly auto-optimize max_length BEFORE we build final input
    # (We also do a check after building input_ids, below.)
    # -- We'll do the final check after the input is built to be safe.

    from .inference import Codec_model, whisper_turbo_pipe

    # Handle reference audio if needed
    speech_ids_prefix = []
    prompt_text = ""
    if generation_mode == "Reference audio" and ref_audio_path:
        progress(0, "Loading & trimming reference audio...")
        waveform, sample_rate = torchaudio.load(ref_audio_path)
        if trim_audio and (waveform.shape[1] / sample_rate) > 15:
            waveform = waveform[:, :sample_rate * 15]

        # Resample to 16k
        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform
        prompt_wav = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform_mono)

        # Transcribe with Whisper
        whisper_args = {}
        if whisper_language != "auto":
            whisper_args["language"] = whisper_language
        prompt_text = whisper_turbo_pipe(prompt_wav[0].numpy(), generate_kwargs=whisper_args)['text'].strip()

        # Encode reference audio with XCodec2
        with torch.no_grad():
            vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0, 0, :]  # shape: [T]
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
    elif generation_mode == "Reference audio" and not ref_audio_path:
        gr.warning("No reference audio provided. Proceeding in text-only mode.")

    progress(0.5, "Generating speech...")

    # Combine any reference text + user text
    combined_input_text = prompt_text + " " + target_text
    prefix_str = "".join(speech_ids_prefix) if speech_ids_prefix else ""
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{combined_input_text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + prefix_str},
    ]
    num_beams = 2 if beam_search_enabled else 1
    early_stopping_val = (num_beams > 1)

    model_inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        return_tensors="pt",
        continue_final_message=True
    )
    input_ids = model_inputs.to("cuda")
    attention_mask = torch.ones_like(input_ids).to("cuda")

    # Final auto-optimize check
    if auto_optimize_length:
        input_len = input_ids.shape[1]
        margin = 100 if generation_mode == "Reference audio" else 50
        if input_len + margin > max_length:
            old_val = max_length
            max_length = input_len + margin
            print(f"Auto optimizing: input length is {input_len}, raising max_length from {old_val} to {max_length}.")

    # Generate tokens
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            max_length=int(max_length),
            min_length=int(max_length * 0.5),
            eos_token_id=speech_end_id,
            do_sample=True,
            num_beams=num_beams,
            length_penalty=1.5,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.2,
            early_stopping=early_stopping_val,
            no_repeat_ngram_size=3,
        )

        prefix_len = len(speech_ids_prefix)
        # cutting off prefix from the final output
        generated_ids = outputs[0][(input_ids.shape[1] - prefix_len) : -1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_tokens = extract_speech_ids(speech_tokens)
        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
        gen_wav = Codec_model.decode_code(speech_tokens)

        # If we had a reference prompt, remove that segment from the final
        if speech_ids_prefix:
            gen_wav = gen_wav[:, :, prompt_wav.shape[1] :]

    sr = 16000
    out_audio_np = gen_wav[0, 0, :].cpu().numpy()
    progress(0.9, "Finalizing audio...")

    audio_data_url = generate_audio_data_url(out_audio_np, sample_rate=sr)
    new_entry = {
        "mode": generation_mode,
        "text": target_text,
        "audio_url": audio_data_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "seed": chosen_seed,
    }

    if len(prev_history) >= MAX_HISTORY:
        prev_history.pop(0)
    prev_history.append(new_entry)
    updated_dashboard_html = render_previous_generations(prev_history, is_generating=False)

    return (sr, out_audio_np), updated_dashboard_html, prev_history


def infer_podcast(
    conversation_text,
    generation_mode,  # "Podcast"
    model_choice,
    hf_api_key,
    trim_audio,
    max_length,
    temperature,
    top_p,
    whisper_language,
    user_seed,
    random_seed_each_gen,
    beam_search_enabled,
    auto_optimize_length,
    prev_history,
    speaker_config=None,
    progress=gr.Progress()
):
    """
    Generate podcast audio line by line, taking speaker-specific configurations.
    """
    if speaker_config is None:
        speaker_config = {}

    from .inference import parse_conversation, generate_audio_data_url, render_previous_generations
    from .inference import join_audio_segments, infer

    conversation, speakers = parse_conversation(conversation_text)
    audio_segments = []

    for speaker, line_text in conversation:
        # Retrieve speaker-specific config
        config = speaker_config.get(speaker.lower(), {"ref_audio": "", "seed": None})
        ref_audio = config.get("ref_audio", "")
        seed = config.get("seed", None)

        # Decide generation mode
        line_mode = "Reference audio" if ref_audio else "Text only"
        result = infer(
            line_mode,
            ref_audio,
            line_text,
            model_choice,
            hf_api_key,
            trim_audio,
            max_length,
            temperature,
            top_p,
            whisper_language,
            seed,
            random_seed_each_gen,
            beam_search_enabled,
            auto_optimize_length,
            prev_history=[],
            progress=progress
        )
        _, line_audio = result[0]
        audio_segments.append(line_audio)

    final_audio = join_audio_segments(audio_segments, sample_rate=16000, crossfade_duration=0.05)
    audio_url = generate_audio_data_url(final_audio, sample_rate=16000)

    new_entry = {
        "mode": "Podcast",
        "text": conversation_text,
        "audio_url": audio_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "seed": "N/A",
    }
    if len(prev_history) >= MAX_HISTORY:
        prev_history.pop(0)
    prev_history.append(new_entry)
    updated_dashboard_html = render_previous_generations(prev_history, is_generating=False)

    return (16000, final_audio), updated_dashboard_html, prev_history
