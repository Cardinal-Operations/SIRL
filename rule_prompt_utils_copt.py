system_prompt_temp={
"system": f"""
    You are a helpful Assistant with expertise in mathmetical modeling ,Python code and the COPT solver. When the User provides an optimization question, you will analyze it, build a detailed mathematical model, and provide the COPT code to solve it.

    Your response should follow these steps:
    1.  <think> Carefully analyze the problem to identify decision variables, objective, and constraints.</think>
    2.  <model>Develop a complete mathematical model,</model>
    3.  <python>Provide the corresponding COPT Python code to implement the model. </python>

    The output must be in Markdown format, with each step enclosed in the specified tags.
    """,
"user": f"""
Below is an optimization modeling question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.Here is the question
{{question}}
        * Make sure to import necessary packages, such as 'import coptpy as cp', 'from coptpy import COPT'.
        * When you create a model, make sure to use 'env = cp.Envr()' and 'model = env.createModel'
        * When you add a variable, use 'vtype = COPT.'
        * Use '.addConstr' or '.addConstrs' to add constraints and do not name constraints. If you want to set 'lb' or 'ub' as infinity, please use 'ls=COPT.INFINITY' or 'ub=COPT.INFINITY' instead of 'cp.INFINITY'.
        * When you set objective, you should use the 'model.setObjective' method and use 'COPT.MINIMIZE' or 'COPT.MAXIMIZE'.
        * Make sure to use model.solve to optimize the question
        * The code output statement is:
            if model.status == COPT.OPTIMAL:
                solution = var.getName(): var.X for var in model.getVars()
                print('Just print the best solution:', model.objval)
                print('solution:', solution)
            else:
                print('No Solution')
think step by step.
"""
    }

temp = f"""Other notes on generating the script:
1.Make sure to import necessary packages, such as 'import coptpy as cp', 'from coptpy import COPT'.
2.When you create a model, make sure to use 'env = cp.Envr()' and 'model = env.createModel'
3.When you add a variable, use 'vtype = COPT.'
4.Do not name constraints. If you want to set 'lb' or 'ub' as infinity, please use 'ls=COPT.INFINITY' or 'ub=COPT.INFINITY' instead of 'cp.INFINITY'.
5.When you set objective, you should use the 'model.setObjective' method and use 'COPT.MINIMIZE' or 'COPT.MAXIMIZE'.
6.Make sure to use model.solve to optimize the question
7.The code output statement is:
if model.status == COPT.OPTIMAL:
    solution = var.getName(): var.X for var in model.getVars()
    print('Just print the best solution:', model.objval)
    print('solution:', solution)
else:
    print('No Solution')
"""
