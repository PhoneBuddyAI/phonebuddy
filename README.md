# PhoneBuddy

**Training open models for agentic phone use with real-app and mock-app environments.**

PhoneBuddy studies how to train open phone-use agents that can complete tasks on real phones. The project compares a shared SFT checkpoint, real-app RL, and mixed real+mock RL using PhoneWorld-style mock apps as scalable, resettable, and automatically verifiable training environments.

## Links

- Project page: https://phonebuddyai.github.io
- Paper PDF: https://phonebuddyai.github.io/assets/paper.pdf
- Code and models: coming soon
- Dataset / benchmark artifacts: coming soon

## Phone-Agent Research Line

PhoneBuddy is part of a broader phone-agent research stack covering environments, model training, runtime execution, privacy, and safety.

| Layer | Project | Links | Role |
| --- | --- | --- | --- |
| Training | **PhoneBuddy** | [Project](https://phonebuddyai.github.io) · [Paper](https://phonebuddyai.github.io/assets/paper.pdf) | Trains open phone-use models with real-app RL and scalable mock-app training. |
| Environment | **PhoneWorld** | [Paper](https://arxiv.org/abs/2605.29486) · [中文Blog](https://mp.weixin.qq.com/s/uzasS6q6LAwX8wLXD7KzeA) | Turns real GUI trajectories and screenshots into controllable phone-use environments, tasks, verifiers, and rollouts. |
| Runtime | **PhoneHarness** | [Project](https://phoneharness.github.io/) · [Paper](https://phoneharness.github.io/assets/paper.pdf) · [Code](https://github.com/PhoneHarness/phoneharness) · [Dataset](https://huggingface.co/datasets/PhoneHarness/phoneharness-bench) · [机器之心](https://mp.weixin.qq.com/s/I2ztL6sFiHGxAiCfh_FTqg?scene=1) | Mixed-action phone-agent harness and benchmark across CLI, GUI, and MCP tools with trace-backed verification. |
| Privacy | **PhonePrivacy / MyPhoneBench** | [Paper](https://arxiv.org/abs/2604.00986) · [中文Blog](https://mp.weixin.qq.com/s/0uqLRepCABA7ptOAPXjDZA) | Verifiable privacy benchmark for phone-use agents. |
| Safety | **PhoneSafety** | [Paper](https://arxiv.org/abs/2605.07630) · [Code](https://github.com/tangzhy/PhoneSafety) | Safety evaluation for phone-use agents, separating safety from incapability. |

## What This Repository Will Contain

- Training and evaluation notes for the PhoneBuddy model line.
- Scripts and documentation for reproducing the public evaluation setup when released.
- Paper figures and project assets.
- Links to model checkpoints, datasets, and benchmark artifacts once available.

## Current Paper Snapshot

The current paper draft reports three checkpoints from the same 4B phone-use model line:

- `PhoneBuddy-4B-SFT`: shared supervised fine-tuning baseline.
- `PhoneBuddy-4B-Real`: continued training with real-app RL.
- `PhoneBuddy-4B-Real+Mock`: mixed RL across real-app and mock-app environments.

Main finding: real-app RL provides realism, while mock-app training adds scalable, resettable, and verifiable interaction signal. The combined recipe improves task success on both real-phone human evaluation and AndroidWorld, with the clearest gains on single-app and mini-app tasks.

## Repository Layout

```text
phonebuddy/
├── assets/
│   ├── figures/      # Paper figures copied from the current manuscript
│   └── paper.pdf     # Current paper snapshot
├── docs/             # Public documentation drafts
└── README.md
```

## Citation

Citation metadata will be added when the public preprint is finalized.
