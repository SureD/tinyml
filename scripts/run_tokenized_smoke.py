from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyinfer_prompt import (
    DEFAULT_MODEL_DIR,
    DEFAULT_RUNNER,
    encode_prompt,
    load_tokenizer,
    parse_token_line,
    run_cpp,
)


DEFAULT_PROMPT = "Hello, my name is"


def decode_tokens(tokenizer, token_ids: list[int], skip_special_tokens: bool) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a TinyInfer smoke test using the real TinyLlama tokenizer."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--show-special-tokens", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_dir)
    prompt_tokens = encode_prompt(tokenizer, args.prompt, add_bos=not args.no_bos)
    if not prompt_tokens:
        raise RuntimeError("tokenizer produced an empty prompt")
    if len(prompt_tokens) + args.max_new_tokens > args.max_seq_len:
        raise RuntimeError("prompt length plus max_new_tokens exceeds max_seq_len")

    print("prompt_text:", args.prompt)
    print("prompt_tokens:", " ".join(str(token) for token in prompt_tokens))
    print(
        "prompt_decoded:",
        decode_tokens(tokenizer, prompt_tokens, skip_special_tokens=not args.show_special_tokens),
    )

    try:
        runner_output = run_cpp(
            args.runner,
            args.model_dir,
            args.max_seq_len,
            args.max_new_tokens,
            prompt_tokens,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    generated_tokens = parse_token_line(runner_output, "generated_tokens")
    all_tokens = parse_token_line(runner_output, "all_tokens")
    generated_text = decode_tokens(
        tokenizer,
        generated_tokens,
        skip_special_tokens=not args.show_special_tokens,
    )
    all_text = decode_tokens(
        tokenizer,
        all_tokens,
        skip_special_tokens=not args.show_special_tokens,
    )

    print(runner_output, end="")
    print("generated_text:", generated_text)
    print("all_text:", all_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
