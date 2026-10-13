# GRPO Reward Granularity for Tool-Using LLMs

A small, hands-on reproduction of the **reward granularity** idea from the paper **[ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)**.

This project uses **Qwen2.5-0.5B-Instruct**, **TRL**, **GRPO**, and **LoRA** to study one simple question:

> If an LLM makes a partially correct tool call, is it better to give it partial credit or only an all-or-nothing reward?

The experiment is intentionally small enough to run in Google Colab, while still showing the mechanics behind reinforcement learning for tool use.

---

## Why this matters

Modern LLMs are increasingly used as agents. They can call search engines, databases, calculators, APIs, code interpreters, and MCP tools.

A tool-using model has to make several decisions:

```text
Do I need a tool?
      ↓
Which tool should I use?
      ↓
Which arguments does it need?
      ↓
What values should those arguments contain?
      ↓
Did the call work?
      ↓
Should I call another tool or stop?
```

This is often called **Tool-Integrated Reasoning (TIR)**: reasoning and tool interaction happen together rather than as separate stages.

Supervised fine-tuning can teach a model to imitate example tool calls, but reinforcement learning is useful when we want the model to learn which behavior works better across multiple possible actions.

The hard part is the reward.

Suppose the correct call is:

```json
{
  "tool": "get_weather",
  "arguments": {
    "city": "Chicago",
    "day": "tomorrow"
  }
}
```

Now suppose the model predicts:

```json
{
  "tool": "get_weather",
  "arguments": {
    "city": "Chicago",
    "day": "today"
  }
}
```

This response is mostly correct. A binary reward gives it the same score as a completely unrelated call. Fine-grained reward can tell the model that the format, tool, schema, and city were correct while the day was wrong.

That is the core intuition tested here.

---

## Connection to MCP

The **Model Context Protocol (MCP)** makes it easier for agents to access tools exposed by different servers and systems.

As the number of tools grows, dynamic tool selection becomes important. A model may need to decide:

- whether a tool is necessary,
- which MCP server contains the right capability,
- which tool to call,
- how to fill its arguments,
- how to recover from tool failures,
- and when to stop calling tools.

Reinforcement learning is a natural fit because these are sequential decisions that can be scored based on outcomes.

This repo focuses on a smaller subproblem: **how reward granularity affects learning correct tool calls**.

---

## Paper inspiration

This experiment is inspired by:

**ToolRL: Reward is All Tool Learning Needs**  
Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji.

- Paper: https://arxiv.org/abs/2504.13958
- PDF: https://arxiv.org/pdf/2504.13958
- Official code: https://github.com/qiancheng0/ToolRL

The original paper studies reward design for tool use across four dimensions:

1. **Type** — what behavior is rewarded?
2. **Scale** — how much does each reward component matter?
3. **Granularity** — how detailed is the feedback?
4. **Dynamics** — should reward weights change during training?

This project concentrates mainly on **granularity**.

---

## Experiment design

### Base model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

A 0.5B model was chosen so the experiment can be run cheaply in Colab.

### Tools

The synthetic environment uses four tools:

```python
calculator(expression)
get_weather(city, day)
lookup_population(city)
convert_temperature(value, unit)
```

Example prompt:

```text
What is 18 multiplied by 4?
```

Expected call:

```json
{
  "tool": "calculator",
  "arguments": {
    "expression": "18*4"
  }
}
```

### Why synthetic tools?

They make reward computation deterministic and easy to inspect. The goal here is not benchmark performance; it is understanding the RL mechanics.

---

## Reward variants

Three models are trained from the same base checkpoint.

### 1. Coarse reward

All-or-nothing:

```text
Exact tool call → 1
Anything else   → 0
```

This is simple but gives poor credit assignment.

### 2. Intermediate reward

```text
Valid JSON                    +0.2
Correct tool                  +0.3
Entire argument set correct   +0.5
```

This separates tool choice from argument correctness.

### 3. Fine-grained reward

```text
Valid JSON              +0.2
Correct tool            +0.3
Correct argument names  +0.2
Correct argument values +0.3
```

The key difference is partial credit.

For example, if the model selects the correct tool and correct argument names but gets one argument value wrong, it still receives useful feedback.

---

## What is GRPO?

**GRPO** stands for **Group Relative Policy Optimization**.

For each prompt, the model generates multiple candidate completions. In this experiment:

```python
num_generations = 4
```

Conceptually:

```text
Prompt
 ├── completion A → reward 0.2
 ├── completion B → reward 1.0
 ├── completion C → reward 0.5
 └── completion D → reward 0.7
```

GRPO compares responses relative to other responses in the same group and increases the probability of higher-reward behavior.

Unlike PPO, GRPO does not require a separate critic/value model for this comparison.

---

## Why LoRA?

Instead of updating every parameter in Qwen, this experiment uses **LoRA** through PEFT.

Conceptually:

```text
Base Qwen model
      +
small trainable LoRA adapter
      =
updated tool-use behavior
```

This makes it practical to train three controlled variants:

```text
              Qwen base
             /    |     \
          Fine  Intermediate  Coarse
```

Each run starts from the same base model.

---

## Libraries used

### Transformers

Loads the Qwen model and tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

### TRL

Provides `GRPOTrainer` and `GRPOConfig`.

```python
from trl import GRPOTrainer, GRPOConfig
```

### PEFT

Provides LoRA adapters.

```python
from peft import LoraConfig, PeftModel
```

### Datasets

Used to create and split the synthetic dataset.

```python
from datasets import Dataset
```

### PyTorch

Used for model execution, GPU handling, and inference.

```python
import torch
```

### pandas / matplotlib

Used for evaluation tables and plots.

---

## Important implementation detail: cleaning tool arguments

One subtle bug appeared when the argument dictionaries were converted into a Hugging Face Dataset.

An intended target like:

```python
{"city": "Los Angeles"}
```

could become:

```python
{
    "city": "Los Angeles",
    "expression": None,
    "day": None,
    "value": None,
    "unit": None
}
```

If this is not cleaned, both the reward and evaluator can incorrectly treat valid predictions as wrong.

The fix is:

```python
def clean_arguments(args):
    if args is None:
        return {}

    return {
        key: value
        for key, value in args.items()
        if value is not None
    }
```

This function should be used inside both reward computation and evaluation.

---

## Another implementation detail: semantic normalization

A calculator may generate:

```text
18 * 4
```

while the target contains:

```text
18*4
```

These are equivalent for the purpose of this experiment.

The evaluator normalizes calculator expressions before comparing them:

```python
def values_equal(pred, gold, key=None):
    if key == "expression":
        return str(pred).replace(" ", "") == str(gold).replace(" ", "")

    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return abs(float(pred) - float(gold)) < 1e-6

    return str(pred).strip().lower() == str(gold).strip().lower()
```

This prevents superficial formatting differences from being counted as reasoning errors.

---

## Core training configuration

A representative GRPO configuration is:

```python
training_args = GRPOConfig(
    output_dir="/content/toolrl_fine",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    num_generations=4,
    generation_batch_size=4,
    max_completion_length=128,
    temperature=1.0,
    top_p=1.0,
    loss_type="grpo",
    scale_rewards="group",
    epsilon=0.2,
    gradient_checkpointing=True,
    use_vllm=False,
    logging_steps=5,
    report_to="none",
    seed=42,
)
```

Then each experiment changes only the reward function:

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=train_small,
    reward_funcs=fine_reward,
    peft_config=peft_config,
)

trainer.train()
```

The same pattern is used for intermediate and coarse rewards.

---

## Evaluation

Four models are compared:

```text
Base Qwen
Coarse GRPO
Intermediate GRPO
Fine GRPO
```

The same held-out examples and deterministic decoding settings are used for all four.

### Metrics

#### Valid JSON

Did the model produce parseable JSON?

#### Tool accuracy

Did it choose the correct tool?

#### Argument-name accuracy

Did it use exactly the required argument names?

#### Argument-value accuracy

Were the required argument values correct?

#### Exact tool-call accuracy

Were the tool name, argument names, and argument values all correct?

This is the strictest headline metric.

---

## Results

Add the corrected final run here after retraining:

| Model | Valid JSON | Tool Accuracy | Argument Names | Argument Values | Exact Tool Call |
|---|---:|---:|---:|---:|---:|
| Base | TBD | TBD | TBD | TBD | TBD |
| Coarse GRPO | TBD | TBD | TBD | TBD | TBD |
| Intermediate GRPO | TBD | TBD | TBD | TBD | TBD |
| Fine GRPO | TBD | TBD | TBD | TBD | TBD |

Recommended chart location:

```text
results/exact_tool_call_accuracy.png
```

---

## How this differs from the original ToolRL paper

| Aspect | Original ToolRL | This repo |
|---|---|---|
| Goal | Full reward-design study | Small educational reproduction |
| Models | Larger Qwen/Llama variants | Qwen2.5-0.5B-Instruct |
| Data | 4K tool-use examples | Small synthetic dataset |
| Tool setting | Realistic benchmark tool calls | Four deterministic tools |
| RL stack | veRL / TinyZero ecosystem | Hugging Face TRL |
| Fine-tuning | Research-scale RL | LoRA |
| Hardware | Multi-GPU research setup | Google Colab GPU |
| Reward dimensions | Type, scale, granularity, dynamics | Mainly granularity |
| Evaluation | External tool-use benchmarks | Held-out synthetic test set |

The goal is therefore not to compare absolute benchmark scores. The useful comparison is whether the qualitative trend appears again: **does richer reward feedback produce better tool-call learning than binary reward?**

---

## Repository structure

```text
toolrl-grpo-reward-granularity/
├── README.md
├── requirements.txt
├── src/
│   ├── rewards.py
│   └── evaluate.py
├── notebooks/
│   └── README.md
└── results/
    └── .gitkeep
```

---

## Install

```bash
pip install -r requirements.txt
```

For Colab, a fresh GPU runtime is recommended.

---

## Suggested next experiments

This project can be extended in several directions:

1. Reward dynamics: change format vs correctness weights over training.
2. Tool necessity: reward the model for avoiding unnecessary calls.
3. Tool cost: penalize expensive or redundant calls.
4. Tool failure recovery: introduce failing APIs and reward successful retry strategies.
5. Larger tool catalogs: study dynamic tool retrieval before tool selection.
6. MCP hierarchy: route from capability → MCP server → tool.
7. Unseen tools: test whether reward-trained behavior generalizes to tools not seen during training.

---

## Main takeaway

A coarse reward says:

```text
Wrong.
```

A fine-grained reward can say:

```text
Your JSON is valid.
The tool is correct.
The argument schema is correct.
One argument value is wrong.
```

For structured tool use, that extra information gives reinforcement learning a much better credit-assignment signal.

---

## References

- **ToolRL: Reward is All Tool Learning Needs**  
  https://arxiv.org/abs/2504.13958

- **Official ToolRL repository**  
  https://github.com/qiancheng0/ToolRL

- **Qwen2.5**  
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

- **TRL**  
  https://github.com/huggingface/trl

- **PEFT**  
  https://github.com/huggingface/peft

---

## Disclaimer

This repository is an independent educational reproduction inspired by ToolRL. It is not the official ToolRL implementation and does not claim benchmark equivalence with the original paper.
