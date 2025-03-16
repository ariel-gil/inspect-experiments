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
        metadata = {"reasonning": reasoning.strip()}, 
    )

def sample_to_fewshot(sample): 
    return(
        f"{sample.input}\n\nReasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER" {sample.target}
    )
