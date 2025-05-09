import gradio as gr
import torch
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from decord import VideoReader, cpu
from holmesvau.holmesvau_utils import load_model, generate
import os
import numpy as np
from PIL import Image

mllm_path = 'HolmesVAU-2B'
sampler_path = './holmesvau/ATS/anomaly_scorer.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)

def save_keyframes_to_folder(vr, idx_list, folder_path="keyframes"):
    os.makedirs(folder_path, exist_ok=True)
    saved_files = []

    for i, idx in enumerate(idx_list):
        frame = vr[idx].asnumpy()  # [H,W,3] ndarray
        img = Image.fromarray(frame)
        filename = os.path.join(folder_path, f"frame_{i}.png")
        img.save(filename)
        saved_files.append(filename)

    return saved_files  

def analyze_video(video_file, prompt):
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            with open(video_file, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name

        pred, history, frame_indices, anomaly_score = generate(
            tmp_path, prompt, model, tokenizer, generation_config, sampler,
            select_frames=12, use_ATS=True
        )

        vr = VideoReader(tmp_path, ctx=cpu(0), num_threads=1)

        keyframe_folder = "keyframes"
        keyframe_files = save_keyframes_to_folder(vr, frame_indices, keyframe_folder)

        plot_path = "anomaly_plot.png"

        if anomaly_score is not None and len(anomaly_score) > 0:
            try:
                plt.figure(figsize=(8, 2))
                plt.plot(anomaly_score, label='Anomaly Score')
                for idx in frame_indices:
                    plt.vlines(idx / 16, 0, 1, colors='r', linestyle="--", linewidth=1)
                plt.ylim(0, 1)
                plt.xlabel('Snippet Index')
                plt.ylabel('Anomaly Score')
                plt.title("Anomaly Score Curve")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            except Exception as e:
                print(f"[ERROR] Plotting failed: {e}")
                plot_path = "error_plot.png"
        else:
            print("[INFO] Uniform Sampling detected - generating flat anomaly plot.")
            dummy_score = np.zeros(len(frame_indices))

            plt.figure(figsize=(8, 2))
            plt.plot(dummy_score, label='Uniform Sampled')
            for i in range(len(frame_indices)):
                plt.vlines(i, 0, 1, colors='gray', linestyle="--", linewidth=1)
            plt.ylim(0, 1)
            plt.xlabel('Snippet Index')
            plt.ylabel('Anomaly Score')
            plt.title("Uniform Sampling (No Anomaly Score)")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        return str(pred), plot_path, keyframe_files
    except Exception as e:
        return f"[ERROR] {str(e)}", "error_plot.png", []

with gr.Blocks(title="HolmesVAU Video Anomaly Detector") as demo:
    gr.Markdown("## 📹 HolmesVAU - Video Anomaly Detection")

    with gr.Row():
        video_input = gr.Video(label="🎥 Upload a video")
        prompt_input = gr.Textbox(label="💬 Prompt", value="Could you specify the anomaly events present in the video?")

    with gr.Row():
        submit_btn = gr.Button("🚀 Run Inference")

    with gr.Row():
        output_text = gr.Textbox(label="🧠 HolmesVAU Prediction")
        output_plot_img = gr.Image(label="📈 Anomaly Score Plot")

    with gr.Row():
        frame_slider = gr.Slider(label="🖼️ Keyframe Index", minimum=0, maximum=11, step=1, visible=False)
    with gr.Row():
        keyframe_img = gr.Image(label="🔍 Selected Keyframe", visible=False)

    keyframe_files_state = gr.State([])  

    def update_keyframe(idx, keyframe_files):
        if 0 <= idx < len(keyframe_files):
            return keyframe_files[idx]
        return None

    def run_and_init_slider(video, prompt):
        pred, plot_path, keyframe_files = analyze_video(video, prompt)
        return (
            pred,                        
            plot_path,                  
            keyframe_files,             
            gr.update(visible=True, maximum=len(keyframe_files)-1),
            gr.update(visible=True, value=keyframe_files[0])
        )

    submit_btn.click(
        fn=run_and_init_slider,
        inputs=[video_input, prompt_input],
        outputs=[output_text, output_plot_img, keyframe_files_state, frame_slider, keyframe_img]
    )

    frame_slider.change(
        fn=update_keyframe,
        inputs=[frame_slider, keyframe_files_state],
        outputs=keyframe_img
    )

if __name__ == "__main__":
    demo.launch()
