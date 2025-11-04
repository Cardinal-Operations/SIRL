import os
import time
import pandas as pd
import json
from vllm import SamplingParams
from content_utils import *
from os.path import splitext, basename
"""
The load data
"""
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
 
def strlen_stat(data):
    lengths = []
    for jsonl in data:
        think = extract_block(jsonl,'think')
        model = extract_block(jsonl,'model')
        code = extract_block(jsonl,'python')
        lengths.append([len(think) if think else 0, len(model) if model else 0, len(code) if code else 0])

    # 转为 DataFrame
    df = pd.DataFrame(lengths, columns=['think_len', 'model_len', 'code_len'])

    # 统计特征
    summary = df.describe(percentiles=[0.25, 0.5, 0.75])[['think_len', 'model_len', 'code_len']].loc[['min', '25%', '50%', '75%', 'max']]
    print(summary)
    print()





def write_string_to_python_file(filepath, string_content, overwrite=False):
    """Writes a string to a Python (.py) file.

    Args:
        filepath: The path to the output .py file.
        string_content: The string to be written to the file.
        overwrite: If True, overwrites the file if it exists. If False, appends to the file. Defaults to False.

    Returns:
        True if the write operation was successful, False otherwise.  Also returns False if the file exists and overwrite is False.
        Prints informative messages about success or failure.
    """

    try:
        mode = 'w' if overwrite else 'a'  # 'w' for write (overwrite), 'a' for append
        if not overwrite and os.path.exists(filepath):
            print(f"File '{filepath}' already exists. Appending to the file.") # Informative message if appending
        with open(filepath, mode, encoding='utf-8') as f: # Handle encoding
            f.write(string_content)
        print(f"Successfully wrote/appended to '{filepath}'.")
        return True
    except Exception as e:
        print(f"An error occurred while writing to '{filepath}': {e}")
        return False

def generate_with_api(client, prompt: str, modelname: str,completion_kwargs:dict):
    messages = [{"role": "user", "content": prompt}]
    if(modelname=='deepseek-v3'):
        modelname = 'deepseek-chat'
    response = client.chat.completions.create(
            messages=messages,
            model=modelname,
            **completion_kwargs
        )
    result_text = str(response.choices[0].message.content)
    return result_text

def save_to_markdown(text,  filepath, filename):
    """Saves a string to a Markdown (.md) file.

    Args:
        text: The input string.
        filepath: The full path to the output Markdown file (including the .md extension).
    """
    filepath = os.path.join(filepath, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as md_file:
            md_file.write(text)

        print(f"String saved to {filepath}")

    except Exception as e:
        print(f"Error creating Markdown file: {e}")


