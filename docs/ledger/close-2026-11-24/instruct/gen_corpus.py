"""The 600 prompt/response records of the #218 instruct-checkpoint check
(seeded, standard library only): python3 gen_corpus.py > pr.jsonl"""
import json
import random
import sys

random.seed(7)
things = ["compiler", "runtime", "plan", "arena", "kernel", "tokenizer",
          "checkpoint", "validator", "optimizer", "adapter", "corpus",
          "gradient"]
verbs = ["allocate", "verify", "load", "bind", "commit", "measure", "lower",
         "seal"]
for i in range(600):
    a, b, v = random.choice(things), random.choice(things), random.choice(verbs)
    sys.stdout.write(json.dumps({
        "prompt": f"Question {i}: how does the {a} {v} the {b}?",
        "response": f"The {a} will {v} the {b} once, at load, and record "
                    f"run {i} in its report."}) + "\n")
