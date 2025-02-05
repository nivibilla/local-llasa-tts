import gradio as gr
import torch
import os

from .inference import (
    initialize_models,
    infer,
    infer_podcast,
    render_previous_generations
)
from .models import check_model_in_cache

# Inline CSS for the dark theme (unchanged)
NEW_CSS = """
/* Remove Gradio branding/footer */
#footer, .gradio-container a[target="_blank"] { display: none; }
/* Simple dark background */
body, .gradio-container { margin: 0; padding: 0; background-color: #1E1E2A; color: #EAEAEA; font-family: 'Segoe UI', sans-serif; }
/* Header styling */
#header { background-color: #2E2F46; padding: 1rem 2rem; text-align: center; }
#header h1 { margin: 0; font-size: 2rem; }
/* Main content row styling */
#content-row { display: flex; flex-direction: row; gap: 1rem; padding: 1rem 2rem; }
/* Synthesis panel */
#synthesis-panel { flex: 2; background-color: #222233; border-radius: 8px; padding: 1.5rem; }
/* History panel */
#history-panel { flex: 1; background-color: #222233; border-radius: 8px; padding: 1.5rem; }
/* Form elements styling */
.gr-textbox input, .gr-textbox textarea, .gr-dropdown select { background-color: #38395A; border: 1px solid #4A4B6F; color: #F1F1F1; border-radius: 4px; padding: 0.5rem; }
/* Audio components */
.audio-input, .audio-output { background-color: #2E2F46 !important; border-radius: 8px !important; padding: 12px !important; margin: 8px 0 !important; }
"""

def build_dashboard():
    """
    Build the Gradio interface with separate tabs for Standard TTS and Podcast Mode.
    Adds a dynamic check for whether the chosen model is in local cache.
    If not, the HF API Key field is shown. If in cache, it remains hidden.
    """
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter")],
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        background_fill_primary="#1E1E2A",
        background_fill_secondary="#222233",
        border_color_primary="#4A4B6F",
        body_text_color="#EAEAEA",
        block_title_text_color="#EAEAEA",
        block_label_text_color="#EAEAEA",
        input_background_fill="#38395A",
    )

    with gr.Blocks(theme=theme, css=NEW_CSS) as demo:
        gr.Markdown("<div id='header'><h1>Llasa TTS Dashboard</h1></div>", elem_id="header")
        # Shared state for previous generations
        prev_history_state = gr.State([])

        ########################################
        # DYNAMIC VISIBILITY: HF API Key Input
        ########################################
        def toggle_api_key_visibility(model_choice):
            in_cache = check_model_in_cache(model_choice)
            # If the model is in local cache, hide the text field
            return gr.update(visible=not in_cache)

        with gr.Tabs():
            # --- Standard TTS Tab ---
            with gr.TabItem("Standard TTS"):
                with gr.Row(elem_id="content-row"):
                    with gr.Column(elem_id="synthesis-panel"):
                        gr.Markdown("## Standard TTS")
                        model_choice_std = gr.Dropdown(
                            label="Select llasa Model", 
                            choices=["1B", "3B", "8B"], 
                            value="3B"
                        )
                        generation_mode_std = gr.Radio(
                            label="Generation Mode", 
                            choices=["Text only", "Reference audio"], 
                            value="Text only", 
                            type="value"
                        )
                        with gr.Group():
                            ref_audio_input = gr.Audio(
                                label="Reference Audio (Optional)", 
                                sources=["upload", "microphone"], 
                                type="filepath"
                            )
                            trim_audio_checkbox_std = gr.Checkbox(
                                label="Trim Reference Audio to 15s?", 
                                value=False
                            )
                        gen_text_input = gr.Textbox(
                            label="Text to Generate", 
                            lines=4, 
                            placeholder="Enter text here..."
                        )

                        with gr.Accordion("Advanced Generation Settings", open=False):
                            max_length_slider_std = gr.Slider(
                                minimum=64, 
                                maximum=4096, 
                                value=1024, 
                                step=64, 
                                label="Max Length (tokens)"
                            )
                            temperature_slider_std = gr.Slider(
                                minimum=0.1, 
                                maximum=2.0, 
                                value=1.0, 
                                step=0.1, 
                                label="Temperature"
                            )
                            top_p_slider_std = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=1.0, 
                                step=0.05, 
                                label="Top-p"
                            )
                            whisper_language_std = gr.Dropdown(
                                label="Whisper Language (for reference audio)",
                                choices=["en", "auto", "ja", "zh", "de", "es", "ru", "ko", "fr"],
                                value="en", 
                                type="value"
                            )
                            random_seed_checkbox_std = gr.Checkbox(
                                label="Random seed each generation", 
                                value=True
                            )
                            beam_search_checkbox_std = gr.Checkbox(
                                label="Enable beam search", 
                                value=False
                            )
                            auto_optimize_checkbox_std = gr.Checkbox(
                                label="[Text Only] Auto Optimize Length", 
                                value=True
                            )
                            seed_number_std = gr.Number(
                                label="Seed (if not random)", 
                                value=None, 
                                precision=0, 
                                minimum=0, 
                                maximum=2**32-1, 
                                step=1
                            )
                        api_key_input_std = gr.Textbox(
                            label="Hugging Face API Key (Required only if model not in cache)", 
                            type="password", 
                            placeholder="Enter your HF token or leave blank",
                            visible=False
                        )

                        synthesize_btn_std = gr.Button("Synthesize")

                        with gr.Group():
                            audio_output_std = gr.Audio(
                                label="Synthesized Audio", 
                                type="numpy", 
                                interactive=False, 
                                show_label=True, 
                                autoplay=False
                            )

                    with gr.Column(elem_id="history-panel"):
                        gr.Markdown("## Previous Generations")
                        dashboard_html_std = gr.HTML(
                            value="<div style='color: #999; font-style: italic;'>No previous generations yet.</div>", 
                            show_label=False
                        )

            # --- Podcast Mode Tab ---
            with gr.TabItem("Podcast Mode"):
                with gr.Row(elem_id="content-row"):
                    with gr.Column(elem_id="synthesis-panel"):
                        gr.Markdown("## Podcast Mode")
                        gr.Markdown("⚠️ **Experimental Feature** ⚠️\nWorks best with reference audio for each speaker.")
                        
                        model_choice_pod = gr.Dropdown(
                            label="Select llasa Model", 
                            choices=["1B", "3B", "8B"], 
                            value="3B"
                        )
                        podcast_transcript = gr.Textbox(
                            label="Podcast Transcript",
                            lines=6,
                            placeholder="Each line -> 'Speaker Name: message'"
                        )
                        with gr.Accordion("Speaker Configuration (Add as many as needed)", open=True):
                            gr.Markdown("Fill out details for each speaker present in the transcript.")
                            speaker1_name = gr.Textbox(
                                label="Speaker 1 Name", 
                                placeholder="e.g., Alex"
                            )
                            ref_audio_speaker1 = gr.Audio(
                                label="Reference Audio for Speaker 1 (Optional)", 
                                sources=["upload", "microphone"], 
                                type="filepath"
                            )
                            seed_speaker1 = gr.Number(
                                label="Seed for Speaker 1 (Optional)", 
                                value=None, 
                                precision=0
                            )
                            
                            speaker2_name = gr.Textbox(
                                label="Speaker 2 Name", 
                                placeholder="e.g., Jamie"
                            )
                            ref_audio_speaker2 = gr.Audio(
                                label="Reference Audio for Speaker 2 (Optional)", 
                                sources=["upload", "microphone"], 
                                type="filepath"
                            )
                            seed_speaker2 = gr.Number(
                                label="Seed for Speaker 2 (Optional)", 
                                value=None, 
                                precision=0
                            )
                            
                            speaker3_name = gr.Textbox(
                                label="Speaker 3 Name (Optional)", 
                                placeholder="e.g., Casey"
                            )
                            ref_audio_speaker3 = gr.Audio(
                                label="Reference Audio for Speaker 3 (Optional)", 
                                sources=["upload", "microphone"], 
                                type="filepath"
                            )
                            seed_speaker3 = gr.Number(
                                label="Seed for Speaker 3 (Optional)", 
                                value=None, 
                                precision=0
                            )

                        with gr.Accordion("Advanced Generation Settings", open=False):
                            max_length_slider_pod = gr.Slider(
                                minimum=64, 
                                maximum=4096, 
                                value=1024, 
                                step=64, 
                                label="Max Length (tokens)"
                            )
                            temperature_slider_pod = gr.Slider(
                                minimum=0.1, 
                                maximum=2.0, 
                                value=1.0, 
                                step=0.1, 
                                label="Temperature"
                            )
                            top_p_slider_pod = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=1.0, 
                                step=0.05, 
                                label="Top-p"
                            )
                            whisper_language_pod = gr.Dropdown(
                                label="Whisper Language (for reference audio)",
                                choices=["en", "auto", "ja", "zh", "de", "es", "ru", "ko", "fr"],
                                value="en", 
                                type="value"
                            )
                            random_seed_checkbox_pod = gr.Checkbox(
                                label="Random seed each generation", 
                                value=True
                            )
                            beam_search_checkbox_pod = gr.Checkbox(
                                label="Enable beam search", 
                                value=False
                            )
                            auto_optimize_checkbox_pod = gr.Checkbox(
                                label="[Text Only] Auto Optimize Length", 
                                value=True
                            )
                            seed_number_pod = gr.Number(
                                label="Seed (if not random)", 
                                value=None, 
                                precision=0, 
                                minimum=0, 
                                maximum=2**32-1, 
                                step=1
                            )

                        api_key_input_pod = gr.Textbox(
                            label="Hugging Face API Key (Required only if model not in cache)", 
                            type="password",
                            placeholder="Enter your HF token or leave blank",
                            visible=False
                        )

                        synthesize_btn_pod = gr.Button("Synthesize Podcast")

                        with gr.Group():
                            audio_output_pod = gr.Audio(
                                label="Synthesized Podcast Audio", 
                                type="numpy", 
                                interactive=False, 
                                show_label=True, 
                                autoplay=False
                            )

                    with gr.Column(elem_id="history-panel"):
                        gr.Markdown("## Previous Generations")
                        dashboard_html_pod = gr.HTML(
                            value="<div style='color: #999; font-style: italic;'>No previous generations yet.</div>", 
                            show_label=False
                        )

        # Define helper callback for Standard TTS
        def synthesize_standard(
            generation_mode, ref_audio_input, gen_text_input, model_choice, api_key_input,
            max_length_slider, temperature_slider, top_p_slider, whisper_language,
            seed_number, random_seed_checkbox, beam_search_checkbox, auto_optimize_checkbox,
            trim_audio, prev_history
        ):
            return infer(
                generation_mode,
                ref_audio_input,
                gen_text_input,
                model_choice,
                api_key_input,
                trim_audio,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                whisper_language,
                seed_number,
                random_seed_checkbox,
                beam_search_checkbox,
                auto_optimize_checkbox,
                prev_history
            )

        # Define helper callback for Podcast
        def synthesize_podcast_fn(
            podcast_transcript, model_choice, api_key_input,
            max_length_slider, temperature_slider, top_p_slider, whisper_language,
            seed_number, random_seed_checkbox, beam_search_checkbox, auto_optimize_checkbox,
            prev_history,
            speaker1_name, ref_audio_speaker1, seed_speaker1,
            speaker2_name, ref_audio_speaker2, seed_speaker2,
            speaker3_name, ref_audio_speaker3, seed_speaker3
        ):
            # Build speaker_config dictionary
            speaker_config = {}
            for name, ref, seed in [
                (speaker1_name, ref_audio_speaker1, seed_speaker1),
                (speaker2_name, ref_audio_speaker2, seed_speaker2),
                (speaker3_name, ref_audio_speaker3, seed_speaker3),
            ]:
                if name and name.strip():
                    speaker_config[name.strip().lower()] = {
                        "ref_audio": ref if ref else "",
                        "seed": seed
                    }

            return infer_podcast(
                podcast_transcript,
                "Podcast",
                model_choice,
                api_key_input,
                False,  # trim_audio
                max_length_slider,
                temperature_slider,
                top_p_slider,
                whisper_language,
                seed_number,
                random_seed_checkbox,
                beam_search_checkbox,
                auto_optimize_checkbox,
                prev_history,
                speaker_config=speaker_config
            )

        # --- Wire up Standard TTS Tab ---
        synthesize_btn_std.click(
            lambda history: render_previous_generations(history, is_generating=True),
            inputs=[prev_history_state],
            outputs=[dashboard_html_std]
        ).then(
            synthesize_standard,
            inputs=[
                generation_mode_std, 
                ref_audio_input, 
                gen_text_input, 
                model_choice_std, 
                api_key_input_std,
                max_length_slider_std, 
                temperature_slider_std, 
                top_p_slider_std, 
                whisper_language_std,
                seed_number_std, 
                random_seed_checkbox_std, 
                beam_search_checkbox_std, 
                auto_optimize_checkbox_std,
                trim_audio_checkbox_std, 
                prev_history_state
            ],
            outputs=[audio_output_std, dashboard_html_std, prev_history_state]
        )

        # --- Wire up Podcast Mode Tab ---
        synthesize_btn_pod.click(
            lambda history: render_previous_generations(history, is_generating=True),
            inputs=[prev_history_state],
            outputs=[dashboard_html_pod]
        ).then(
            synthesize_podcast_fn,
            inputs=[
                podcast_transcript, 
                model_choice_pod, 
                api_key_input_pod,
                max_length_slider_pod, 
                temperature_slider_pod, 
                top_p_slider_pod, 
                whisper_language_pod,
                seed_number_pod, 
                random_seed_checkbox_pod, 
                beam_search_checkbox_pod, 
                auto_optimize_checkbox_pod,
                prev_history_state,
                speaker1_name, ref_audio_speaker1, seed_speaker1,
                speaker2_name, ref_audio_speaker2, seed_speaker2,
                speaker3_name, ref_audio_speaker3, seed_speaker3
            ],
            outputs=[audio_output_pod, dashboard_html_pod, prev_history_state]
        )

        # Show/hide API key input if model not cached
        model_choice_std.change(
            toggle_api_key_visibility, 
            inputs=[model_choice_std], 
            outputs=[api_key_input_std]
        )
        model_choice_pod.change(
            toggle_api_key_visibility, 
            inputs=[model_choice_pod], 
            outputs=[api_key_input_pod]
        )

        # On load, also run the toggle once to set correct visibility
        demo.load(
            fn=toggle_api_key_visibility,
            inputs=[model_choice_std],
            outputs=[api_key_input_std]
        )
        demo.load(
            fn=toggle_api_key_visibility,
            inputs=[model_choice_pod],
            outputs=[api_key_input_pod]
        )

    return demo
