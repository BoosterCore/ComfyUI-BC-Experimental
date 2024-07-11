import os
import re
from typing import List
from .utils.replace_tokens import replace_tokens
import folder_paths
from folder_paths import get_output_directory
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
import hashlib
import requests
import json


# 同时带有正反向提示词输入框的ClipTextEncode节点
class ClipTextEncodeBC:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
            },
        }

    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING',)
    RETURN_NAMES = ('CONDITIONING_P', 'CONDITIONING_N',)
    FUNCTION = "encode"

    CATEGORY = "🚀 BCE Nodes/conditioning"

    def encode(self, clip, positive, negative,):
        tokens_p = clip.tokenize(positive)
        cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
        tokens_n = clip.tokenize(negative)
        cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
        return (
            [[cond_p, {"pooled_output": pooled_p}]],
            [[cond_n, {"pooled_output": pooled_n}]],
        )


# 同时带有两个正向及一个反向提示词输入框的ClipTextEncode节点
class ClipTextEncodeBCA:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positiveA": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "positiveB": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, }),
            },
        }

    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING', 'CONDITIONING',)
    RETURN_NAMES = ('CONDITIONING_PA', 'CONDITIONING_PB', 'CONDITIONING_N',)
    FUNCTION = "encode"

    CATEGORY = "🚀 BCE Nodes/conditioning"

    def encode(self, clip, positiveA, positiveB, negative,):
        tokens_pa = clip.tokenize(positiveA)
        cond_pa, pooled_pa = clip.encode_from_tokens(tokens_pa, return_pooled=True)
        tokens_pb = clip.tokenize(positiveB)
        cond_pb, pooled_pb = clip.encode_from_tokens(tokens_pb, return_pooled=True)
        tokens_n = clip.tokenize(negative)
        cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
        return (
            [[cond_pa, {"pooled_output": pooled_pa}]],
            [[cond_pb, {"pooled_output": pooled_pb}]],
            [[cond_n, {"pooled_output": pooled_n}]],
        )


# 可以定义任意后缀及输出路径的文本保存节点
class SaveAnyText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_text": ("STRING", {"forceInput": True}),
                "path": ("STRING", {"default": '[time(%Y-%m-%d)]/', "multiline": False}),
                "prefix": ("STRING", {"default": "[time(%Y-%m-%d - %H.%M.%S)]"}),
                "counter_separator": ("STRING", {"default": "_"}),
                "counter_length": ("INT", {"default": 3, "min": 0, "max": 24, "step": 1}),
                "suffix": ("STRING", {"default": ""}),
                "output_extension": ("STRING", {"default": "txt"})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_full_path", "output_name")
    FUNCTION = "save_any_text"
    CATEGORY = "🚀 BCE Nodes/IO"

    def save_any_text(self, file_text, path, prefix='[time(%Y-%m-%d %H.%M.%S)]', counter_separator='_',
                       counter_length=3, suffix='', output_extension='txt'):
        path = replace_tokens(path)
        prefix = replace_tokens(prefix)
        suffix = replace_tokens(suffix)

        # Safety check to ensure the extension is not empty
        if not output_extension.strip():
            raise ValueError("The output extension cannot be empty.")

        # Safety check to prevent directory traversal
        if '..' in path or any(esc in path for esc in ['..\\', '../']):
            raise ValueError(
                "The specified path contains invalid characters that navigate outside the output directory.")

        # Check if the path is absolute
        if os.path.isabs(path):
            full_path = os.path.abspath(path)
        else:
            # Get the base output directory from folder_paths
            output_base_dir = get_output_directory()
            full_path = os.path.join(output_base_dir, path)
            full_path = os.path.abspath(full_path)

            # Ensure the path is within the allowed directory
            if not full_path.startswith(output_base_dir):
                raise ValueError("The specified path is outside the allowed output directory")

        if not os.path.exists(full_path):
            print(f"Warning: The path `{full_path}` doesn't exist! Creating it...")
            try:
                os.makedirs(full_path, exist_ok=True)
            except OSError as e:
                print(f"Error: The path `{full_path}` could not be created! Is there write access?\n{e}")

        if file_text.strip() == '':
            raise ValueError("There is no text specified to save! Text is empty.")

        separator = counter_separator
        number_padding = int(counter_length)
        file_extension = f'.{output_extension}'

        filename, counter = self.generate_filename(full_path, prefix, suffix, separator, number_padding, file_extension)
        file_path = os.path.join(full_path, filename)

        # Remove extension from output_name
        output_name = os.path.splitext(filename)[0]

        self.writeTextFile(file_path, file_text)

        return file_path, output_name

    def generate_filename(self, path, prefix, suffix, separator, number_padding, extension):
        """Generate a unique filename based on the provided parameters."""
        pattern_parts = [re.escape(prefix)]
        if number_padding > 0:
            pattern_parts.append(f"{separator}(\\d{{{number_padding}}})")
        if suffix:
            pattern_parts.append(f"{separator}{re.escape(suffix)}")
        pattern_parts.append(re.escape(extension))

        pattern = ''.join(pattern_parts)

        # Find existing counters
        existing_counters = []
        for filename in os.listdir(path):
            match = re.match(pattern, filename)
            if match:
                try:
                    counter_value = int(match.group(1)) if number_padding > 0 else 0
                    existing_counters.append(counter_value)
                except (IndexError, ValueError):
                    continue

        existing_counters.sort(reverse=True)

        # Determine the next counter value
        counter = existing_counters[0] + 1 if existing_counters else 1
        counter_str = f"{counter:0{number_padding}}" if number_padding > 0 else ""

        # Construct the filename
        if number_padding > 0:
            if suffix:
                filename = f"{prefix}{separator}{counter_str}{separator}{suffix}{extension}"
            else:
                filename = f"{prefix}{separator}{counter_str}{extension}"
        else:
            filename = f"{prefix}{suffix}{extension}"

        return filename, counter

    def writeTextFile(self, file, content):
        """Write the content to the specified file."""
        try:
            with open(file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
        except OSError as e:
            print(f"Unable to save file `{file}`: {e}")
            raise


# 简易的多行文本输入节点
class SimpleText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": "", })
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"

    CATEGORY = "🚀 BCE Nodes/text"

    def func(self,string):
        return (
            string,
        )


# 带有Trigger Word提取及作用开关的LoRA加载节点
def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None


def save_dict_to_json(data_dict, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to file: {e}")


def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class LoraWithTriggerWord:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP",),
                             "lora_name": (LORA_LIST,),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                             "query_tags": ("BOOLEAN", {"default": True}),
                             "tags_out": ("BOOLEAN", {"default": True}),
                             "print_tags": ("BOOLEAN", {"default": False}),
                             "force_fetch": ("BOOLEAN", {"default": True}),
                             "enable": ("BOOLEAN", {"default": True}),
                             },
                "optional":
                    {
                        "opt_prompt": ("STRING", {"forceInput": True}),
                    }
                }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"
    CATEGORY = "🚀 BCE Nodes/loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, query_tags, tags_out, print_tags,
                  force_fetch, enable, opt_prompt=None):
        if strength_model == 0 and strength_clip == 0 or enable == 0:
            if opt_prompt is not None:
                out_string = opt_prompt
            else:
                out_string = ""
            return (model, clip, out_string,)

        json_tags_path = "./loras_tags.json"
        lora_tags = load_json_from_file(json_tags_path)
        output_tags = lora_tags.get(lora_name, None) if lora_tags is not None else None
        if output_tags is not None:
            output_tags = ", ".join(output_tags)
            if print_tags:
                print("trainedWords:", output_tags)
        else:
            output_tags = ""

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if (query_tags and output_tags == "") or force_fetch:
            print("calculating lora hash")
            LORAsha256 = calculate_sha256(lora_path)
            print("requesting infos")
            model_info = get_model_version_info(LORAsha256)
            if model_info is not None:
                if "trainedWords" in model_info:
                    print("tags found!")
                    if lora_tags is None:
                        lora_tags = {}
                    lora_tags[lora_name] = model_info["trainedWords"]
                    save_dict_to_json(lora_tags, json_tags_path)
                    output_tags = ", ".join(model_info["trainedWords"])
                    if print_tags:
                        print("trainedWords:", output_tags)
            else:
                print("No informations found.")
                if lora_tags is None:
                    lora_tags = {}
                lora_tags[lora_name] = []
                save_dict_to_json(lora_tags, json_tags_path)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        if opt_prompt is not None:
            if tags_out:
                output_tags = opt_prompt + ", " + output_tags
            else:
                output_tags = opt_prompt
        return (model_lora, clip_lora, output_tags,)