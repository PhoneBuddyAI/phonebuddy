# PhoneBuddy

**Training open models for agentic phone use with real-app and mock-app environments.**

PhoneBuddy studies how to train open phone-use agents that can complete tasks on real phones. The project compares a shared SFT checkpoint, real-app RL, and mixed real+mock RL using PhoneWorld-style mock apps as scalable, resettable, and automatically verifiable training environments.

## Links

- Project page: https://phonebuddyai.github.io
- Paper PDF: https://phonebuddyai.github.io/assets/paper.pdf
- Code and models: coming soon
- Dataset / benchmark artifacts: coming soon

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

