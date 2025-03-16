from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate



## Eval 
@task 
def security_guide(): 
    return Task(
        dataset = [Sample(
            input = "just reply with hello world", 
            target = "hello world",
            )
            ],
        solver = [generate()], ## using a specific solver
        scorer = exact(),
        )
