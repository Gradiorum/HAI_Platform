import re

def parse_output_for_answer(output):
    pattern = r"<Tag>\[(.*?)\]</Tag>"
    matches = re.findall(pattern, output)
    return matches
