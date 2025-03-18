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
        match = re.search(AnswerPattern.LINE, state.output.completion) ## checking for answer format in AnswerPattern  #anki re.
        if match: 
            ## Clean up \boxed {}
            clean_target = re.sub()
            # ask model to judge equivalnce 
            answer = match.group(1)
            prompt = EQUIVALENCE_TEMPLATE % ({"expression1": target.text, "expression2": answer}) # anki - % string formatting. also .format(), %(string)s
            result = await get_model().generate(prompt)

            ## Return score
            correct = result.completion.lower().strip() == "yes"
            return Score(
                value=CORRECT if correct else INCORRECT,
                answer=answer, 
                explanation=state.output.completion, 
                )
        else: 
            return Score(
                value=INCORRECT, 
                explanation="Answer not found in model output: " + f"{state.output.completion}", 
            )
    return score


## anki r"""

EQUIVALENCE_TEMPLATE = r""" 
Look at the following two expressions (answers to a math problem)
and judge whether they are equivalent. Only perform trivial 
simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: \boxed{0}
    Expression 2: 0

Yes
(ignore \boxed{} or other LaTeX notation)

---

YOUR TASK

Respond with only "Yes" or "No" (without quotes). Do not include
a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

