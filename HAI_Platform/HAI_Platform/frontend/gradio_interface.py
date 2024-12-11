import gradio as gr
import requests

API_URL = "http://0.0.0.0:8000"

def chat_submit(prompt, model_name, method, rl_method):
    resp = requests.post(f"{API_URL}/chat", json={"prompt": prompt, "model_name": model_name, "method": method, "user_id":"ui_user"})
    data = resp.json()
    return data["response"], data["interaction_id"]

def send_feedback(interaction_id, feedback_type, details):
    resp = requests.post(f"{API_URL}/feedback", json={"interaction_id": interaction_id, "feedback_type": feedback_type, "details": details, "user_id":"ui_user"})
    return resp.json()

def build_interface():
    with gr.Blocks() as demo:
        with gr.Tab("Chat Mode"):
            model_select = gr.Dropdown(["gpt-4", "llama-2", "custom-model"], label="Model")
            method_select = gr.Dropdown(["none", "dense_verifier", "adaptive_distribution", "entropix", "mcts", "deepseek"], label="Test-time Compute Method")
            rl_mode = gr.Dropdown(["none", "kto", "ppo", "dpo"], label="RLHF Mode")
            prompt_box = gr.Textbox(label="Prompt")
            submit_btn = gr.Button("Submit")
            outputs_box = gr.Textbox(label="Model Output", interactive=False)
            hidden_interaction_id = gr.Variable()

            def on_submit(prompt, model_name, method, rl_method):
                resp, interaction_id = chat_submit(prompt, model_name, method, rl_method)
                return resp, interaction_id

            submit_btn.click(on_submit, [prompt_box, model_select, method_select, rl_mode], [outputs_box, hidden_interaction_id])

            feedback_type = gr.Radio(["accept", "reject"], label="Feedback")
            feedback_btn = gr.Button("Submit Feedback")

            def on_feedback(feedback_value, interaction_id):
                return send_feedback(interaction_id, feedback_value, "User feedback")

            feedback_btn.click(on_feedback, [feedback_type, hidden_interaction_id], None)

        with gr.Tab("Data Import & Batch"):
            gr.Markdown("Upload CSV and select columns...")

        with gr.Tab("RLHF Data Collection"):
            gr.Markdown("PPO / DPO Interface...")

        with gr.Tab("Research Mode"):
            gr.Markdown("Eye Tracking / EEG integration...")

        with gr.Tab("Settings"):
            gr.Markdown("Adjust model params, templates, etc.")

    return demo

if __name__ == "__main__":
    ui = build_interface()
    ui.launch(server_name="0.0.0.0", server_port=7860)
