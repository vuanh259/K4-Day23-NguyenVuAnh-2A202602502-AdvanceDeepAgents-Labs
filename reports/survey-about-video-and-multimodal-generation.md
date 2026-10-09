# Video and Multimodal Generation: From Video Diffusion to Interactive World Models

## TL;DR
- Video generation developed from image-diffusion extensions toward systems that combine spatial synthesis with temporal modeling, conditioning, and multimodal context. Early video diffusion work jointly trained on image and video data and supported conditional and unconditional generation [1].
- Recent systems divide into complementary design points: unified diffusion Transformers for several conditioning modes [2], low-latency distilled interactive video diffusion [3], and closed-loop models that generate clips as part of goal-directed action execution [4].
- Evaluation is moving beyond frame quality and basic prompt adherence. VBench++ separates temporal and frame-wise dimensions and adds human-aligned and trustworthiness dimensions [5], while VBench-2.0 targets intrinsic faithfulness, including physics, commonsense, controllability, and human fidelity [6].
- Safety and reliability remain distinct from visual quality. T2VSafetyBench reports heterogeneous safety strengths and a usability-safety trade-off across nine text-to-video models [7], while high-dynamic video understanding tests show that sparse sampling can miss brief causal events [8].

## Background
Video generation asks a model to synthesize a sequence whose frames are individually plausible and collectively coherent. Multimodal generation broadens the conditioning and output space: a system may combine text, images, audio, video, or language-model context, and may generate an avatar, a continuation, a response to an image, or a video-conditioned action sequence. These tasks share a central difficulty: the output must preserve identity, objects, motion, and causal structure over time while following an input instruction.

A foundational step was to treat video diffusion as a natural extension of image diffusion. *Video Diffusion Models* jointly trained on image and video data, and its indexed summary describes conditional sampling for video prediction and unconditional generation [1]. This framing established a useful separation between spatial generation and temporal structure. It also made video a natural target for diffusion methods already successful in images. The limitation of this early paradigm is conceptual as well as computational: a generated clip can look good frame by frame without faithfully representing motion, physical interaction, or the intended event sequence.

The field now contains several overlapping families rather than one settled architecture. Diffusion remains prominent, but distillation, in-context conditioning, multimodal encoders, and closed-loop planning change what “video generation” means. The resulting systems should therefore be compared by conditioning, temporal scope, interaction loop, and evaluation target, not by a single quality score.

## 1. Diffusion as a foundation for temporal synthesis

The early video-diffusion approach extends image generation into space and time and uses joint image/video training to make optimization practical [1]. Its significance is not merely that it produces moving images. It establishes a reusable recipe: initialize from image-generation knowledge, add temporal capacity, and use conditioning to predict or extend video. This recipe supports multiple settings, including unconditional generation, video prediction, and text-conditioned synthesis [1].

That flexibility also reveals a persistent trade-off. A short fixed-length clip can be modeled as a large spatiotemporal object, but longer or interactive video requires continuation, autoregression, or another mechanism for retaining context. Temporal extension can preserve local continuity while still drifting in identity or scene state. Conversely, stronger global conditioning can improve instruction following while making the system more expensive or less responsive. These are inferences from the different system designs and evaluation targets in the retrieved evidence, rather than a directly measured universal law.

## 2. Unified and multimodally conditioned generation

Recent work broadens video generation from text prompts to several input modalities. SkyReels-V3 presents a unified multimodal in-context-learning framework with diffusion Transformers for reference-image-to-video, video extension, and audio-guided talking-avatar generation [2]. The retrieved description also reports cross-frame pairing, image editing, semantic rewriting, hybrid image/video training, and multi-resolution optimization aimed at reference adherence [2]. Its value is architectural breadth: one model family addresses generation, continuation, editing, and audio-driven performance rather than treating each as an isolated task.

LiveTalk emphasizes a different objective. It uses improved on-policy distillation and optimized scheduling for real-time multimodal video generation, with text, image, and audio context [3]. The reported system integrates audio-language models and identity-focused mechanisms for interactive avatars [3]. Its abstract reports comparable visual quality to similar or larger full-step bidirectional baselines at 20 times lower inference cost and latency on HDTF, AVSpeech, and CelebV-HQ, and reports higher coherence and content quality than Sora2 and Veo3 on a curated multi-turn benchmark [3]. These are reported results from the source, not independent replication.

The comparison illustrates two important axes. SkyReels-V3 prioritizes coverage of conditioning and generation modes [2], while LiveTalk prioritizes responsiveness and repeated interaction [3]. Neither description establishes that one dominates the other: the retrieved SkyReels-V3 evidence does not provide named benchmark scores, and LiveTalk’s comparisons concern particular baselines and tasks. Multimodal conditioning also increases evaluation burden. A model must follow semantic instructions, preserve a reference identity, synchronize audio and motion, and maintain coherence across turns.

## 3. From open-loop clips to goal-directed world models

A further shift treats generated video as an intermediate representation for acting. WorldGuide takes an initial image and a task goal, predicts an atomic action, renders a clip, and then chooses the next action or stops using generated visual progress [4]. Its closed-loop ContextPlanner/Executor design initializes from Qwen2.5-VL-7B and HunyuanVideo-1.5, and its WorldGuide Bench contains approximately 59,000 step-annotated videos across 245 tasks and 27 categories [4].

This is not simply a stronger text-to-video model. It combines action-conditioned generation, visual-feedback planning, termination prediction, and hierarchical visual memory [4]. Its reported metric is task success: 33.33% versus 29.90% for MiniMax-H3 on WorldGuide Bench, where the competitor receives reference action plans, and 47.69% versus 32.73% on VideoCraft-Bench under goal-only conditioning [4]. Because competitor conditioning differs on the first benchmark, those numbers should not be interpreted as a clean generic generation ranking [4].

World models expose a broader definition of correctness. A plausible clip is insufficient if it does not advance the goal, represent the correct action, or terminate appropriately. This direction connects multimodal generation with agents, simulation, and procedural video, but it also makes errors harder to localize: failure may come from planning, rendering, perception, memory, or the interface among them.

## 4. Evaluation beyond visual plausibility

VBench++ argues that existing metrics do not fully align with human perception and evaluates 16 disentangled dimensions for text-to-video and image-to-video [5]. It separates temporal quality from frame-wise quality, includes human preference annotations, and measures temporal flickering and motion smoothness with distinct procedures [5]. Its trustworthiness evaluation covers culture fairness, human bias, and safety; its operational safety procedure uses several frame classifiers and flags a video when more than half of its frames are classified as unsafe [5]. The framework therefore treats temporal coherence, human judgment, and responsible generation as separate evaluation objects.

VBench-2.0 extends this concern to intrinsic faithfulness. Its retrieved description argues that strong per-frame aesthetics, temporal consistency, and basic prompt adherence can coexist with fundamentally unrealistic outputs [6]. It evaluates Human Fidelity, Controllability, Creativity, Physics, and Commonsense using generalist VLM/LLM evaluators and specialist anomaly detectors [6]. This is an important methodological change: the question is no longer only whether a clip looks smooth, but whether its people, actions, physical behavior, and compositional relationships make sense.

Safety requires its own testing. T2VSafetyBench describes four primary categories and 14 critical aspects, drawing prompts from real-world prompts, LLM-generated prompts, and jailbreak attacks [7]. Its evaluation of nine text-to-video models reports that no single model excels across all aspects, and identifies a usability-safety trade-off [7]. These findings caution against collapsing safety into one aggregate quality number.

## Trends and open problems

Work from 2024 through 2026 shows a movement toward broader conditioning, faster inference, longer or continued video, and interaction with an external goal. Evaluation is following that movement: benchmark design increasingly separates dimensions, adds human preference evidence, and tests physical or commonsense faithfulness [5][6]. The newest systems also suggest a convergence between generation and action. WorldGuide uses generated clips inside a feedback loop [4], while LiveTalk treats generation as a low-latency conversational interface [3].

Several problems remain open. First, temporal reliability is still difficult to measure. FastBench reports 306 question-answer pairs across eight domains and six capability dimensions, with human-annotated evidence intervals [8]. The strongest model reached 50.7%; Qwen3-VL-8B improved from 32.9% at 2 FPS to 44.6% at 24 FPS, while performance saturated as history was compressed [8]. These results concern streaming video understanding rather than generation quality, but they expose a related bottleneck: sparse sampling can miss brief causal events, and denser sampling competes with context limits [8]. Generated-video benchmarks need comparable tests of short-lived causes, not just smooth appearance.

Second, benchmark scores remain difficult to compare across conditioning regimes. WorldGuide’s first comparison gives the competitor reference plans while its own setup is goal-only [4]. LiveTalk’s cost and quality claims compare distilled and full-step systems, while its multi-turn comparison uses a curated benchmark [3]. Future reports should state inputs, interaction protocol, temporal horizon, and human or automated judging procedure with the same precision as model architecture.

Third, automated evaluators need calibration. VBench++ and VBench-2.0 use detectors, VLMs, LLMs, and human annotations [5][6], but the retrieved evidence does not provide detector error rates or complete evaluator calibration statistics. Finally, safety and faithfulness must be tested together. A model can satisfy a prompt while producing harmful content, or generate a safe-looking scene that is physically or causally false. Progress therefore requires datasets and protocols that jointly measure identity, motion, instruction following, causal consistency, physical plausibility, bias, and misuse resistance across the full multimodal interaction loop.

## References
[1] Video Diffusion Models. hf-search. https://huggingface.co/papers/2204.03458 (2022-04-07)
[2] SkyReels-V3 Technique Report. hf-search. https://huggingface.co/papers/2601.17323 (2026-01-24)
[3] LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation. hf-search. https://huggingface.co/papers/2512.23576 (2025-12-29)
[4] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. arxiv. https://arxiv.org/abs/2610.12459 (2026-10-08)
[5] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. web. https://arxiv.org/html/2411.13503v1 (unknown)
[6] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. web. https://huggingface.co/papers/2503.21755 (2025-03-27)
[7] T2VSafetyBench: evaluating the safety of text-to-video generative models. web. http://dl.acm.org/doi/10.5555/3737916.3739955 (2024-12-10)
[8] FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?. arxiv. https://arxiv.org/abs/2610.12427 (2026-10-08)
