import os
import re
from typing import List
from .utils.replace_tokens import replace_tokens
from folder_paths import get_output_directory


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
