def rlhf_sample(prompt, chosen, rejected):
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }
