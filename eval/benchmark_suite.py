import torch
import time
import argparse
import json
import itertools

from PIL import Image
import pandas as pd

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

# Ensure reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def benchmark_vlm(
    vit_model_type: str,
    lm_model_type: str,
    lm_tokenizer_path: str,
    mp_pixel_shuffle_factor: int,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
    warmup_runs: int,
    device: torch.device,
):
    """
    Benchmarks a VLM configuration and returns timing and memory metrics.
    """
    # (printing omitted for brevity)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = VLMConfig(
        vit_model_type=vit_model_type,
        lm_model_type=lm_model_type,
        lm_tokenizer=lm_tokenizer_path,
        mp_pixel_shuffle_factor=mp_pixel_shuffle_factor,
        vlm_load_backbone_weights=True
    )
    model = VisionLanguageModel(cfg, load_backbone=True).to(device).eval()
    tokenizer_name = cfg.lm_tokenizer or cfg.lm_model_type
    tokenizer = get_tokenizer(tokenizer_name, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size)

    initial_vram_model_mb = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
        initial_vram_model_bytes = torch.cuda.memory_allocated(device)
        initial_vram_model_mb = initial_vram_model_bytes / (1024 ** 2)

    # Prepare inputs
    template = f"Question: {prompt} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    input_ids = encoded_batch['input_ids'].to(device)
    attention_mask = encoded_batch['attention_mask'].to(device)
    pil_image = Image.open(image_path)
    image_tensor = image_processor(pil_image).unsqueeze(0).to(device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            max_new_tokens=1,
            greedy=True,
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    total_t, peak_mem = [], []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        _ = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_t.append(time.perf_counter() - start)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_mem.append(torch.cuda.max_memory_allocated(device) / (1024**2))
        else:
            peak_mem.append(0)

    avg_total = sum(total_t) / num_runs
    avg_peak = sum(peak_mem)/len(peak_mem) if peak_mem else 0

    # Cleanup
    daresult = {
            "vit_model_type": vit_model_type,
            "lm_model_type": lm_model_type,
            "mp_pixel_shuffle_factor": mp_pixel_shuffle_factor,
            "avg_total_time": avg_total,
            "initial_vram_model_mb": initial_vram_model_mb,
            "avg_peak_vram_inference_mb": avg_peak,
    }
    del model, tokenizer, image_processor
    if device.type == 'cuda': torch.cuda.empty_cache()
    return daresult


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VLM inference speed with JSON logging and analysis.")

    parser.add_argument("--vit_model_types", type=str, nargs='+', default=["google/siglip2-base-patch16-256", "google/siglip2-base-patch16-512", "google/siglip2-so400m-patch16-512"],
                        help="List of ViT model identifiers.")
    parser.add_argument("--lm_model_types", type=str, nargs='+', default=["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B"],
                        help="List of LLM model identifiers.")
    parser.add_argument("--lm_tokenizer", type=str, default="HuggingFaceTB/cosmo2-tokenizer",
                        help="LLM tokenizer identifier.")
    parser.add_argument("--mp_pixel_shuffle_factors", type=int, nargs='+', default=[1, 2, 4],
                        help="List of pixel shuffle factors.")
    parser.add_argument("--image_path", type=str, default="assets/image.png",
                        help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="What is in this image?",
                        help="Prompt for the VLM.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of new tokens to generate.")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of times to run the benchmark.")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs before benchmarking.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_file = 'benchmark_results.json'
    cached_results = {}
    try:
        with open(results_file, 'r') as f:
            existing_results_list = json.load(f)
            for r in existing_results_list:
                # Ensure all necessary keys are present for a valid cached entry
                if all(k in r for k in ['vit_model_type', 'mp_pixel_shuffle_factor', 'lm_model_type']):
                    key = (r['vit_model_type'], r['mp_pixel_shuffle_factor'], r['lm_model_type'])
                    cached_results[key] = r
                else:
                    print(f"Warning: Skipping invalid or incomplete entry in '{results_file}': {r}")
            print(f"Loaded {len(cached_results)} existing valid results from '{results_file}'.")
    except FileNotFoundError:
        print(f"'{results_file}' not found. Starting with an empty cache.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{results_file}'. Starting with an empty cache.")


    # Generate combinations
    all_combinations = list(itertools.product(
        args.vit_model_types,
        args.mp_pixel_shuffle_factors,
        args.lm_model_types
    ))

    # Collect results for this run
    results_for_this_run = []
    for vit, pixel_shuffle, lm in all_combinations:
        current_key = (vit, pixel_shuffle, lm)
        if current_key in cached_results:
            print(f"\nLoading cached result for ViT={vit}, pixel_shuffle={pixel_shuffle}, LLM={lm}")
            res = cached_results[current_key]
        else:
            print(f"\nBenchmarking ViT={vit}, pixel_shuffle={pixel_shuffle}, LLM={lm}")
            res = benchmark_vlm(
                vit_model_type=vit,
                lm_model_type=lm,
                lm_tokenizer_path=args.lm_tokenizer,
                mp_pixel_shuffle_factor=pixel_shuffle,
                image_path=args.image_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                device=device,
            )
            cached_results[current_key] = res # Add/update in our master cache
        results_for_this_run.append(res)

    # Save all known results (including new ones) to JSON
    with open(results_file, 'w') as jf:
        json.dump(list(cached_results.values()), jf, indent=2)
    print(f"\nSaved all (old and new) results to '{results_file}'")

    # Create DataFrame for the combinations processed in this run
    df = pd.DataFrame(list(cached_results.values()))
    print("\n--- Summary Table ---")
    print(df)
