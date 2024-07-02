from .BCE_nodes import ClipTextEncodeBC
from .BCE_nodes import SaveAnyText
from .BCE_nodes import SimpleText
from .BCE_nodes import LoraWithTriggerWord

NODE_CLASS_MAPPINGS = {
    "ClipTextEncodeBC": ClipTextEncodeBC,
    "SaveAnyText": SaveAnyText,
    "SimpleText": SimpleText,
    "LoraWithTriggerWord": LoraWithTriggerWord,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipTextEncodeBC": "🚀 CLIP Text Encode (BCE)",
    "SaveAnyText": "🚀 Save Any Text (BCE)",
    "SimpleText": "🚀 Simple Text (BCE)",
    "LoraWithTriggerWord": "🚀 Lora With Trigger Word (BCE)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
