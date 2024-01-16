# Contributing to FlagPerf

We are happy to accept your contributions to make `FlagPerf` better and more awesome! To avoid unnecessary work on either
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/FlagOpen/FlagPerf/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the
commit guidelines below.

## Sign the CLA

Before you can contribute to FlagPerf, you will need to sign the [Contributor License Agreement](CLA.md).

## Git Commit Guidelines

If there is already a ticket, use this number at the start of your commit message.
Use meaningful commit messages that described what you did.

**Example:** `GH-42: Added new type of embeddings: DocumentEmbedding.`
<br>
**Example:** `ISSUE#175: Alignment of the computational workload for the MindSpore ResNet-50 training case.`


## Development Guide

For contributors looking forward to getting deeper into the project, we suggest cloning the repository and checking out [introduction documents](https://github.com/FlagOpen/FlagPerf/tree/main/docs_zh) and [development documents](https://github.com/FlagOpen/FlagPerf/tree/main/docs/dev). Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.

### add a new pretraining standard case

Please refer to [pretraining standard case specification](https://github.com/FlagOpen/FlagPerf/blob/main/docs/dev/specifications/standard-case-spec.md) for help.

### add a vendor adaption for standard case
Please refer to [standard case adaption specification](https://github.com/FlagOpen/FlagPerf/blob/main/docs/dev/specifications/case-adaption-spec.md) for help.

### add a new inference standard case or add an adaption for inference standard case

Please refer to [inference standard case specification](https://github.com/FlagOpen/FlagPerf/blob/main/docs/dev/inference-case-doc.md) for help.

### bugfix

The recommended approach is to submit [an issue](https://github.com/FlagOpen/FlagPerf/issues) in the code repository to start a disuccsion with the repository maintainers. If it is confirmed to be a bug by us, then submitting a Pull Request (PR) to resolve it is warmly welcomed.

### code formatting

To ensure a standardized code style we use the formatter [yapf](https://github.com/google/yapf).
<br>
You can automatically format the code via **`yapf . -i --recursive`** in the flagperf root folder.
