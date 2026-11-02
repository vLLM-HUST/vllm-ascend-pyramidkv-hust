# Qualification after the standalone release

The [current-version qualification and handoff plan](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/docs/plans/pyramidkv-standalone-qualification-20261103.md)
lives in the canonical benchmark repository. It freezes the standalone candidate,
defines quality, service-cost and capacity work, and records the execution-resource
and calibrated-judge prerequisites.

Implementation is coordinated by @Irisuko through
[issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1) and
[issue #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8).
Candidate-package acceptance does not establish measured effects or provide
resource allocation. Historical shared-adapter results keep their original
source identity. Measured originals and execution plans remain in the benchmark
repository; this repository owns plugin code, integration and links to evidence.
