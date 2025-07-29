
import json
import re
def enforce_integer_variables(code):
    """
    This function modifies the code to ensure that all variables created using
    addVar or addVars are of type GRB.INTEGER.
    """
    # match the addVar or addVars function call
    pattern = r'(\w+\s*=\s*\w+\.addVar[s]?)\(([\s\S]*?)(\)\n)'
    
    def replacer(match):
        var_assignment = match.group(1)  # the variable construction, such as "x = m.addVar" or "x = m.addVars"
        params = match.group(2).rstrip()  # the parameters, such as "lb=0, ub=1, name='x'"
        closing = match.group(3)         # the closing part, such as ")\n"
        
        # if vtype is already present, return the original match
        if re.search(r'\bvtype\s*=', params):
            return match.group(0)
        
        if params:
            # if there are parameters, add vtype=GRB.INTEGER to the end
            if not params.endswith(','):
                params += ','
            new_params = f"{params} vtype=GRB.INTEGER"
        else:
            # if there are no parameters, just add vtype=GRB.INTEGER
            new_params = "vtype=GRB.INTEGER"
        
        # return the modified string
        return f"{var_assignment}({new_params}{closing}"
    
    # replace all matches in the code
    return re.sub(pattern, replacer, code, flags=re.MULTILINE)

def change_variable_types(str_log):
    # check if the log contains vtype
    # if there is a vtype, we need to change the type
    # if it is integer, we change it to continuous
    # if it is continuous, we change it to integer
    if "Vtype" in str_log or "vtype" in str_log:
        if 'INTEGER' in str_log:
            return str_log.replace('INTEGER', 'CONTINUOUS')
        elif 'CONTINUOUS' in str_log:
            return str_log.replace('CONTINUOUS', 'INTEGER')
    # if there is no vtype, we assume the variables are continuous
    # and we need to change them to integer
    else:
        return enforce_integer_variables(str_log)
    
def insert_print(code: str, solver_name: str) -> str:
    # match the model.optimize() or model.solve() line
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)  # get indentation
        model_name = model_match.group(2)  # get model name
        # insert print statement after the model.optimize() or model.solve() line
        if solver_name == "gurobi":
            pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
            status_check = (
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best solution: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        # use re to match the pattern and keep the indent
        elif solver_name == "copt":
            pattern = r'^(\s*)(' + model_name + r'\.solve\(\))'
            status_check = (
                f"{indent}if {model_name}.status == COPT.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)

        
    return code


def extract_code_block(llm_output: str,solver_name) -> str:
    """
    to extract code block
    """
    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if '```' in code:  # the code block may be in ```python ``` format
            pattern = r'```python(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        code = insert_print(code, solver_name)
        return code
    # the python code block is not in <python> </python> format
    # try to extract it using ```python ``` format
    pattern = r'```python(.*?)```'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        code = insert_print(code,solver_name)
        return code
    return None

def extract_obj(str_log,solver_name):
    """Extract objective value from log string"""
    if solver_name == 'gurobi' and 'Just print the best solution:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best solution:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    if solver_name == 'copt' and 'Just print the best obj:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best obj:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None

def load_jsonl(filepath):
    """Loads a JSONL (JSON Lines) file and returns a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    return data
 