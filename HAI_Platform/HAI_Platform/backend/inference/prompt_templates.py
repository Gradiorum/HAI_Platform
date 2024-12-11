import json
import os

with open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "templates_config.json")) as f:
    TEMPLATES = json.load(f)

def build_prompt(model_name, user_prompt, system_msg=None, response_placeholder="{response}"):
    template = TEMPLATES.get(model_name, TEMPLATES["default"])
    system = template.get("system", "")
    user_t = template.get("user_template", "{prompt}")
    assistant_t = template.get("assistant_template", "{response}")

    if system_msg is None:
        system_msg = system

    final_prompt = ""
    if system_msg:
        final_prompt += system_msg + "\n"
    final_prompt += user_t.format(prompt=user_prompt)
    return final_prompt, assistant_t
