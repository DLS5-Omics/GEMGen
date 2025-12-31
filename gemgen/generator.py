import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from utils import read_lines
from nlm_tokenizer import NatureLM1BTokenizer

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    print("Error: PyTorch is required to check CUDA availability.")
    exit(1)

if not torch.cuda.is_available():
    print("CUDA is not available. This program requires a CUDA-enabled GPU to run. Exiting.")
    exit(1)
    

class GEMGenGenerator:
    def __init__(
        self,
        tokenizer_path="",
        ckpt_path="",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.8,
    ):
        """
        SFMGenerator class is used to generate responses for the given input string.
        """

        tokenizer = NatureLM1BTokenizer.from_pretrained(tokenizer_path)
        llm = LLM(
            model=ckpt_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info(f"Built a SFM model from {ckpt_path}.")

        self.model = llm
        self.tokenizer = tokenizer

    def decode_and_process(self, token_ids):
        s = self.tokenizer.decode(token_ids)
        segs = s.split(self.tokenizer.eos_token)
        resp = segs[0].strip()
        if "<mol>" in resp:
            resp = resp.replace("<m>", "").replace("<mol>", "").replace("</mol>", "")
    
        return resp

    def chat_batch(self, input_list,**kwargs):
        """
        Keyword arguments:
            input_list: a list of strings
        Returns:
            output_list: a list of lists of strings [[str, str], [str, str], ...]
        """

        max_new_tokens = kwargs.pop("max_new_tokens", 512)
        temperature = kwargs.pop("temperature", 1)
        top_p = kwargs.pop("top_p", 0.95)
        sample_count = kwargs.pop("sample_count", 3)
        batch_size = kwargs.pop("sample_batch_size", 10)

        logger.info(
            f"Sampling parameters: max_new_tokens: {max_new_tokens},temperature : {temperature},top_p: {top_p},sample_count: {sample_count}"
        )

        input_list = [
            input_str.strip().replace("<enter>", "\n") for input_str in input_list
        ]
        prompt_list = [
            f"Instruction: {input_str.strip()}\n\n\nResponse:"
            for input_str in input_list
        ]


        prompt_token_ids = [
            self.tokenizer(prompt)["input_ids"] for prompt in prompt_list
        ]

        output_list = []
        for i in range(len(prompt_token_ids)):
            output_list.append([])

        pbar = tqdm(
            total=sample_count,
            desc="Generating samples",
            dynamic_ncols=True,
        )
        for i in range(0, sample_count, batch_size):
            current_batch_size = min(batch_size, sample_count - i)
            sampling_params = SamplingParams(
                n=current_batch_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens
            )
            outputs = self.model.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            pbar.update(current_batch_size)
            for j, output in enumerate(outputs):
                cur_out_list = []
                for out in output.outputs:
                    resp = self.decode_and_process(out.token_ids)
                    cur_out_list.append(resp)

                output_list[j].extend(cur_out_list)
            if i == 0:
                print(output_list[0])
        return output_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="VLLM inference for GEMGenGenerator. Input: txt (one prompt per line). Output: json list [[input, [outputs...]], ...]."
    )

    # paths
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer directory/path.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Model checkpoint directory/path for vLLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file. One prompt per line.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path.")

    # vllm parallelism
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM gpu_memory_utilization.")

    # sampling params
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--sample_count", type=int, default=10, help="Number of samples per input.")
    parser.add_argument("--sample_batch_size", type=int, default=10, help="Batch size for sampling.")
    args = parser.parse_args()

    gen = GEMGenGenerator(
        tokenizer_path=args.tokenizer_path,
        ckpt_path=args.ckpt_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    inputs = read_lines(args.input_file)
    if len(inputs) == 0:
        raise ValueError(f"Empty input file: {args.input_file}")


    outputs = gen.chat_batch(
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        sample_batch_size=args.sample_batch_size
    )

    result = []
    for inp, out_list in zip(inputs, outputs):
        result.append([inp, out_list])

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(result)} items to {args.output_file}")
