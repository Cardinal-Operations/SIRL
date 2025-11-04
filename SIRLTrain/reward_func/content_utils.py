import re
import subprocess
import textwrap
# ---------------------------------------
# 提取代码块的函数
# ---------------------------------------

def insert_print(code: str, solver_name: str) -> str:
    # 动态匹配模型名字
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)  # 获取缩进
        model_name = model_match.group(2)  # 获取模型名字
        optimize_call = model_match.group(3)  # 获取优化调用方法
        # 根据求解器名称设置优化调用方法
        if solver_name == "gurobi":
            pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
            status_check = (
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
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
        # 使用正则表达式替换，并保持相同的缩进
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
    return code

def insert_lp_generation(code: str,output_name:str) -> str:
    # 动态匹配模型名字
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    try:
        code = str(code)  # 尝试转换为字符串
    except:
        return None
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)  # 获取缩进
        model_name = model_match.group(2)  # 获取模型名字
        optimize_call = model_match.group(3)  # 获取优化调用方法
        # 根据求解器名称设置优化调用方法
        pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
        status_check = (
            f"{indent}{model_name}.write('{output_name}')\n"
            f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
            f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
            f"{indent}else:\n"
            f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
        )

        # 使用正则表达式替换，并保持相同的缩进
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
    return code

def extract_code_block(llm_output: str,solver_name) -> str:
    """
    使用正则提取三引号 ```python ...``` 之间的代码（DOTALL 模式）。
    若未匹配到则返回空字符串。
    """
    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if '```' in code: #可能python内部额外加了代码块
            pattern = r'```python(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        code = insert_print(code, solver_name)
        return code
    # 可能没有pyhon符号
    pattern = r'```python(.*?)```'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        code = insert_print(code,solver_name)
        return code
    return None

def extract_block(llm_output,part_name):
    # 识别math的部分
    pattern = rf'<{part_name}>(.*?)</{part_name}>'
    block = None
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        block = match.group(1).strip()
    return block

def extract_obj(str_log):
    """Extract objective value from log string"""
    if 'Just print the best obj:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best obj:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None

def extract_sol(str_log):
    """Extract objective value from log string"""
    if 'Just print the best sol:' in str_log:
        sol_match = re.search(r'Just print the best sol:\s*\[([-\d.,\s]*)\]', str_log)
        best_sol = [float(x) for x in sol_match.group(1).split(',') if x.strip()] if sol_match else None
        if best_sol:
            best_sol.sort()
            return best_sol
        else:
            print(str_log)
            return [None]
    return [None]

def extract_integer_binary(str_log):
    """Extract objective value from log string"""
    return 'Integer Variables Exists' in str_log or 'Binary Variables Exists' in str_log
import re

def enforce_integer_variables(code):
    """
    在 Gurobi 的 addVar/addVars 调用中，在最后一个参数后（反括号和换行符前）插入 vtype=GRB.INTEGER。
    允许参数中包含任意字符。
    """
    # 匹配 addVar/addVars 调用，捕获任意参数部分和 )\n
    pattern = r'(\w+\s*=\s*\w+\.addVar[s]?)\(([\s\S]*?)(\)\n)'
    
    def replacer(match):
        var_assignment = match.group(1)  # 变量赋值部分，如 "x = m.addVar" 或 "x = m.addVars"
        params = match.group(2).rstrip()  # 参数部分，去除右侧空格
        closing = match.group(3)         # 闭合括号和换行符，如 ")\n"
        
        # 如果已经有 vtype=，跳过修改
        if re.search(r'\bvtype\s*=', params):
            return match.group(0)
        
        # 处理参数
        if params:
            # 确保在最后一个参数后添加逗号（如果需要）
            if not params.endswith(','):
                params += ','
            new_params = f"{params} vtype=GRB.INTEGER"
        else:
            # 没有参数时，直接添加 vtype
            new_params = "vtype=GRB.INTEGER"
        
        # 返回修改后的字符串，保留闭合括号和换行符
        return f"{var_assignment}({new_params}{closing}"
    
    # 应用替换并返回修改后的代码
    return re.sub(pattern, replacer, code, flags=re.MULTILINE)

def change_variable_types(str_log):
    # 找到有没有vtype字符: 如果有，检查是INTEGER 还是 CONTINUOUS, 替换成另一种
    if "Vtype" in str_log or "vtype" in str_log:
        if 'INTEGER' in str_log:
            return str_log.replace('INTEGER', 'CONTINUOUS')
        elif 'CONTINUOUS' in str_log:
            return str_log.replace('CONTINUOUS', 'INTEGER')
    # 如果没有，说明默认是CONTINUOUS 类型. 找到变量生成模块， 换成INTEGER， 
    else:
        return enforce_integer_variables(str_log)
