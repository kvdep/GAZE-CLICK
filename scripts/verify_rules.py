import re

FORBIDDEN_WORDS = ['шумный', 'шумная', 'шумное', 'шумные', 'фундаментальный', 'фундаментальная', 'фундаментальные', 'энергия', 'энергии']

def check_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    found = []
    for w in FORBIDDEN_WORDS:
        matches = re.findall(rf'\b{w}\b', text, re.IGNORECASE)
        if matches:
            found.extend(matches)
    
    if found:
        print(f"[FAIL] {filename} contains forbidden words: {set(found)}")
    else:
        print(f"[PASS] {filename} is clean of forbidden words.")

check_file('docs/09_ready_models_and_aggregators.md')
check_file('docs/11_detailed_model_architectures.md')
