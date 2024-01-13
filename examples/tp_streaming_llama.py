import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import argparse
import json
import os
import time
import re
import sys

from torch.nn.parameter import Parameter
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
import math
import torch.nn.functional as F

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm

from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from typing import Union, List, Tuple, Optional

def generate_chunk_sizes(tensor_size, num_chunks):
    base_chunk_size = tensor_size // num_chunks
    remainder = tensor_size % num_chunks

    # Create a list of chunk sizes, initially all equal to the base size
    chunk_sizes = [base_chunk_size] * num_chunks

    # Distribute the remainder, starting from the last chunk
    for i in range(0, remainder):
        chunk_sizes[i] += 1

    return chunk_sizes

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    group = torch.distributed.distributed_c10d._get_default_group()
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            if rank == 0:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    
    if rank == 0:
        print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None

    time_list = []
    start_time = time.time()
    for idx, prompt in enumerate(prompts):
        epoch_start_time = time.time()
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        # if kv_cache is not None:
        #     space_needed = seq_len + max_gen_len
        #     past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
        past_key_values = None

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
        elapsed = time.time() - start_time
        time_list.append(elapsed)
        if dist.get_rank() == 0:
            print("| batch {:5d} | time {:5.2f} ".format(
                        idx, elapsed
                    )
                )
        start_time = time.time()
        if idx > 10:
            break
    print(time_list)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, ref):
        super().__init__()

        self.config = ref.config
        self.hidden_size = ref.hidden_size
        self.num_heads = ref.num_heads
        self.head_dim = ref.head_dim
        self.num_key_value_heads = ref.num_key_value_heads
        self.num_key_value_groups = ref.num_key_value_groups
        self.max_position_embeddings = ref.max_position_embeddings
        self.rope_theta = ref.rope_theta

        self.q_proj = ref.q_proj
        self.k_proj = ref.k_proj
        self.v_proj = ref.v_proj
        self.o_proj = ref.o_proj
        self.rotary_emb = ref.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        group = torch.distributed.distributed_c10d._get_default_group()
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
        self.config.pretraining_tp = 2
        
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )[rank]
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)[rank]
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)[rank]

            query_states = F.linear(hidden_states, query_slices)
            key_states = F.linear(hidden_states, key_slices)
            value_states = F.linear(hidden_states, value_slices)
            num_heads = self.num_heads // self.config.pretraining_tp
            num_key_value_heads = self.num_key_value_heads // self.config.pretraining_tp
            hidden_size = self.hidden_size // self.config.pretraining_tp

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            num_heads = self.num_heads

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        if self.config.pretraining_tp > 1:
            # attn_output = attn_output.split(hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(hidden_size, dim=1)[rank]
            # attn_output = F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])

            attn_output = F.linear(attn_output, o_proj_slices)
            torch.distributed.all_reduce(attn_output, group=group)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def RecursiveVisit(name, module, upper_module):
    has_child = any(isinstance(child, nn.Module) for child in module.children())

    if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
        r = LlamaAttention(module)
        setattr(upper_module, name, r)
        return
    if has_child:
        for name, child in module.named_children():
            RecursiveVisit(name, child, module)

def main(rank, args, world_size):
    init_method_pgroup = "tcp://localhost:{}".format(29555)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )
    torch.cuda.set_device(rank)

    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    RecursiveVisit("model", model, model)
    model.cuda()

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())

    
    mp.spawn(
        main,
        args=(args, num_devices),
        nprocs=num_devices,
        join=True,
    )
