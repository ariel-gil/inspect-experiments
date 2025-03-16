from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import(
    generate, prompt_template, system_message
)

## Convert raw records (?) to samples - dividing the "reasoning" and "final answer" segments of the benchmark format. 
## Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10

def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM) ## split() - divide to substrings based on seperator ->DELIM
    target = answer.pop().strip() ## Pop the last element which is the final answer. strip() - remove whitespace
    reasoning = DELIM.join(answer) 
    return Sample(
        input = input, 
        target = target, 
        metadata = {"reasoning": reasoning.strip()}, 
    )

def sample_to_fewshot(sample): 
    return (
        f"{sample.input}\n\nReasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER: {sample.target}"
        )

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem.


{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to 
the problem, and you do not need to use a \\boxed command.

Reasoning: 
""".strip()

@task
def gsm8k(fewshot=10, fewshot_seed=42):
    #build solver list dynamically (may or may not be using fewshot)
    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k", 
            data_dir="main",
            split="train",
            sample_fields=record_to_sample, 
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0, 
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
                ),
            )
    
## define task 
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=solver,
        scorer=match(numeric=True),
    )

