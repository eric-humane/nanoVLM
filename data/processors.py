from transformers import AutoTokenizer
import torchvision.transforms as transforms

from data.custom_transforms import DynamicResize, SplitImage, GlobalAndSplitImages

TOKENIZERS_CACHE = {}


class _DummyTokenizer:
    def __init__(self, extra_special_tokens=None):
        base_specials = {"<eos>": 0, "<pad>": 0}
        self.token_to_id = dict(base_specials)
        self.id_to_token = {v: k for k, v in base_specials.items()}
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        extra_tokens = extra_special_tokens or {}
        for tok in extra_tokens.values():
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        for token_name, token_value in extra_tokens.items():
            setattr(self, token_name, token_value)
            setattr(self, f"{token_name}_id", self.token_to_id[token_value])

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, self.eos_token_id)

    def __len__(self):
        return len(self.token_to_id)

    def batch_decode(self, ids, skip_special_tokens=False):
        return [" ".join(self.id_to_token.get(int(i), "<unk>") for i in seq) for seq in ids]

    def add_special_tokens(self, tokens_dict):
        tokens = tokens_dict.get("additional_special_tokens", [])
        added = 0
        for tok in tokens:
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                added += 1
        return added


def _cache_key(name, extra_special_tokens, chat_template):
    extra_tokens_tuple = None
    if isinstance(extra_special_tokens, dict):
        extra_tokens_tuple = tuple(sorted(extra_special_tokens.items()))
    elif extra_special_tokens is not None:
        extra_tokens_tuple = tuple(extra_special_tokens)
    return (name, extra_tokens_tuple, chat_template)


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    cache_key = _cache_key(name, extra_special_tokens, chat_template)
    if cache_key not in TOKENIZERS_CACHE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        except Exception:
            if name == "testing":
                tokenizer = _DummyTokenizer(extra_special_tokens or {})
            else:
                raise
        if chat_template is not None:
            tokenizer.chat_template = chat_template

        token_list = []
        if isinstance(extra_special_tokens, dict):
            token_list = list(extra_special_tokens.values())
        elif extra_special_tokens is not None:
            token_list = list(extra_special_tokens)

        if token_list:
            tokenizer.add_special_tokens({"additional_special_tokens": token_list})

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Expose special tokens as attributes (e.g., tokenizer.image_token, tokenizer.image_token_id)
        if isinstance(extra_special_tokens, dict):
            for token_name, token_value in extra_special_tokens.items():
                setattr(tokenizer, token_name, token_value)
                token_id = tokenizer.convert_tokens_to_ids(token_value)
                setattr(tokenizer, f"{token_name}_id", token_id)

        TOKENIZERS_CACHE[cache_key] = tokenizer

    return TOKENIZERS_CACHE[cache_key]

def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])

def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:  # If there is only one patch, treat it as the global image
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}')
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string
