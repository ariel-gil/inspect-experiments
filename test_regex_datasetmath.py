import re
from inspect_ai.dataset import hf_dataset, FieldSpec

def test_regex_on_dataset():
    # Load the dataset
    dataset = hf_dataset(
        "nlile/hendrycks-MATH-benchmark", 
        split="test", 
        sample_fields=FieldSpec(
            input="problem",
            target="solution"
        ),
        trust=True
    )

    def clean_solution(solution):
        # Preserve numeric values and degrees
        clean = re.sub(r'(\\boxed|\\text|\\mathbb|\\mathcal){([^}]*)}', r'\2', solution)
        clean = re.sub(r'\$', '', clean)  # Remove dollar signs
        return clean.strip()

    # Iterate directly over the dataset
    for sample in dataset:
        original = sample.target  # Use .target instead of ['solution']
        cleaned = clean_solution(original)
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 50)

# Run the test
test_regex_on_dataset()