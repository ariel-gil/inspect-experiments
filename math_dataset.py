import re
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.model import GenerateConfig, get_model 
from inspect_ai.scorer import CORRECT, INCORRECT, AnswerPattern, Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import(
    generate, prompt_template, TaskState
)
#from datasets import load_dataset
#ds = load_dataset("nlile/hendrycks-MATH-benchmark")

## Defs 


PROMPT_TEMPLATE = """
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
def math(shuffle=True):
    return Task(
        dataset=hf_dataset(
        "nlile/hendrycks-MATH-benchmark", 
        split="test", 
        sample_fields=FieldSpec(
            input="problem",
            target="solution"
        ),
        shuffle=shuffle,
        trust=True
    ), 
    ## implement 
    solver = [
        prompt_template(PROMPT_TEMPLATE),
        generate()
    ], 
    scorer=expression_equivalence(), 
    config=GenerateConfig(temperature=0.5),
    )

## Math expressions can be logically equivalent but not literally the same
## Custom scorer to implement this (using a model) - 

@scorer(metrics=[accuracy(), stderr()])
def expression_equivalence():
    async def score(state: TaskState, target: Target):
        ## extract answer 
        match = re.search(AnswerPattern.LINE, state.output.completion) ## checking for answer format in AnswerPattern 
        if match: 
            # ask model to judge equivalnce 
            answer = match.group(1)
            prompt = EQUIVALENCE_TEMPLATE % ({"expression1": target.text, "expression2:": answer})
            result = await get_model().generate(prompt)

            ## Return score
            correct = result.completion.lower() == "yes"
            return Score(
                value=CORRECT if correct else INCORRECT
                answer=answer, 
                explanation=state.output.completion, 
            )
        else: 
            return Score(
                value=INCORRECT, 
                explanation="Answer not found in model output: " + f"{state.output.completion}", 
            )
    return score



   