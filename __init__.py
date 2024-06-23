from .BCE_nodes import ClipTextEncodeBC
from .BCE_nodes import SaveAnyText
from .BCE_nodes import SimpleText

NODE_CLASS_MAPPINGS = {
    "ClipTextEncodeBC": ClipTextEncodeBC,
    "SaveAnyText": SaveAnyText,
    "SimpleText": SimpleText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipTextEncodeBC": "🚀 CLIP Text Encode (BCE)",
    "SaveAnyText": "🚀 Save Any Text (BCE)",
    "SimpleText": "🚀 Simple Text (BCE)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
