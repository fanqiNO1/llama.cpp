import argparse
from typing import Dict

import numpy as np
import torch
from gguf import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    original_name = name
    name = name.replace("vision_tower", "v").replace("encoder.", "")
    name = name.replace("blocks", "blk")
    # special process for Attention
    if 'wqkv' in name:
        name = name.replace(".wqkv", ".attn_wqkv")
    if 'wo' in name:
        name = name.replace(".wo", ".attn_out")
    
    name = name.replace("mlp.fc0", "ffn_down").replace("mlp.fc1", "ffn_up")
    name = name.replace("norm0", "ln1").replace("norm1", "ln2")
    name = name.replace("multi_modal_projector.linear_", 'mm.')
    name = name.replace("patch_embed.proj.", "patch_embd.")
    name = name.replace("patch_embed.pos_emb", "position_embd")
    print(f"[to_gguf_name] {original_name} --> {name}")
    return name


def find_vision_tensors(kimi_vl, dtype) -> Dict[str, np.ndarray]:
    vision_tower = kimi_vl.vision_tower
    tensor_map = {}
    for name, ten in vision_tower.state_dict().items():
        ten = ten.numpy()
        if 'wqkv' in name:
            if ten.ndim == 2:  # weight
                c3, _ = ten.shape
            else:  # bias
                c3 = ten.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = ten[:c]
            wk = ten[c: c * 2]
            wv = ten[c * 2:]
            tensor_map[to_gguf_name(f"vision_tower.{name}").replace("wqkv", "q")] = wq
            tensor_map[to_gguf_name(f"vision_tower.{name}").replace("wqkv", "k")] = wk
            tensor_map[to_gguf_name(f"vision_tower.{name}").replace("wqkv", "v")] = wv
        elif 'multi_modal_projector' in name:
            if name.endswith("pre_norm.weight"):
                tensor_map['v.post_ln.weight'] = ten
            elif name.endswith("pre_norm.bias"):
                tensor_map['v.post_ln.bias'] = ten
            else:
                # "multi_modal_projector.linear_%d.weight/bias" --> "mm.%d.weight/bias"
                tensor_map[to_gguf_name(name)] = ten
        else:
            tensor_map[to_gguf_name(f"vision_tower.{name}")] = ten
    
    for new_name, ten in tensor_map.items():
        if ten.ndim <= 1 or new_name.endswith("_norm.weight"):
            tensor_map[new_name] = ten.astype(np.float32)
        else:
            tensor_map[new_name] = ten.astype(dtype)
    return tensor_map


def main(args):
    if args.data_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.data_type == 'fp16':
        dtype = torch.float32
        np_dtype = np.float16
        ftype = 1
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}, use fp32 or fp16")
    # Load the model
    local_model = False
    model_path = ""
    model_name = args.model_name
    print(f"model name: {model_name}")
    kimi_vl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    cfg: AutoConfig = kimi_vl.config
    vcfg = cfg.vision_config

    if os.path.isdir(model_name):
        local_model = True
        if model_name.endswith(os.sep):
            model_name = model_name[:-1]
        model_path = model_name
        model_name = os.path.basename(model_name)
    fname_out = f"{model_name.replace('/', '-').lower()}-vision.gguf"

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_description("image encoder for Kimi-VL")

    fout.add_file_type(ftype)
    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_kimi_vl_projector", True)
    fout.add_string("clip.projector_type", "kimi_vl_projector")

    print(f'vision config: {vcfg}')
    # MoonVit uses `PytorchGELUTanh`
    fout.add_bool("clip.use_silu", False)
    fout.add_bool("clip.use_gelu", True)

    tensor_map = find_vision_tensors(kimi_vl, np_dtype)
    for name, data in tensor_map.items():
        fout.add_tensor(name, data)
    
    fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
    fout.add_uint32("clip.vision.image_size", 14 * 64)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vcfg.hidden_size)
    fout.add_uint32("clip.vision.projection_dim", vcfg.intermediate_size)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_attention_heads)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-5)  # torch.nn.LayerNorm default eps
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.num_hidden_layers)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 0)  # not sure what this does, put 0 here as a placeholder
    fout.add_name(model_name)

    if local_model:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    fout.add_array("clip.vision.image_mean", processor.image_processor.image_mean)
    fout.add_array("clip.vision.image_std", processor.image_processor.image_std)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print(f"saved to {fname_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs='?', default="moonshotai/Kimi-VL-A3B-Instruct")
    parser.add_argument("--data_type", nargs='?', choices=['fp32', 'fp16'], default="fp32")
    args = parser.parse_args()
    main(args)

