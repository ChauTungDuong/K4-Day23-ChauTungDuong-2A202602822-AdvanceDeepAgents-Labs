# A Comprehensive Survey on World Models: Architectures, Generative Simulators, and LLM Integration

## TL;DR
- World models serve as internal simulation engines enabling AI agents to predict environmental dynamics, plan long-horizon actions, and learn policies efficiently without raw real-world trial-and-error [1][2].
- Architectural paradigms span Recurrent State-Space Models (RSSM), Joint Embedding Predictive Architectures (JEPA), spatiotemporal transformers, and video generation models [2][3][4].
- Recent breakthroughs scale video generation models (e.g., Sora, Genie) and Foundation Models into interactive physical simulators capable of unsupervised latent action discovery and 3D consistency [4][5][6].
- Large Language Models and Multimodal Foundation Models encode rich implicit world models through next-token prediction and internal reasoning paradigms, supporting task verification and model-based reinforcement learning [7][8][9].
- Key open challenges include mitigating compounding prediction errors, bridging sim-to-real gaps, and ensuring robust spatial-temporal physical reasoning [3][9].

## Background
The paradigm of world models traces its roots back to control theory, model-based reinforcement learning (MBRL), and cognitive psychology, where intelligent agents construct internal representations of external environments to simulate future outcomes before acting [1][2]. In modern artificial intelligence, a world model formalizes an environment's transition dynamics, observation distribution, and reward structures. By shifting learning and planning from high-dimensional raw observation spaces into compact latent or simulated spaces, world models achieve exceptional data efficiency and generalization [2][3].

Recent taxonomy frameworks organize world models across four primary dimensions: architectural design (recurrent vs. transformer vs. diffusion), representation learning objectives (reconstruction vs. contrastive/predictive embedding), reasoning and planning strategies (latent imagination, tree search, policy optimization), and application domains (robotics, autonomous driving, video generation, and text-based agent reasoning) [2][3][10]. The convergence of deep generative modeling, transformer scaling, and foundation models has transformed world models from auxiliary model-based RL components into foundational simulators of physical and virtual reality [4][9].

## Architectural Paradigms and Representation Learning
World model architectures have evolved significantly to balance computational tractability, long-horizon stability, and representational capacity [2][3]. Foundational approaches in model-based reinforcement learning center around the **Recurrent State-Space Model (RSSM)** family (such as PlaNet and the DreamerV1–V3 series) [1][2]. RSSM factorizes environment dynamics into a deterministic recurrent state (capturing long-term temporal memory) and a stochastic component (representing aleatoric uncertainty) [2][3]. While early models relied on continuous Gaussian distributions, DreamerV2 and DreamerV3 established that discrete categorical latent variables (e.g., multi-categorical spaces with high representational capacity) improve performance across diverse discrete and continuous control benchmarks [2].

To bypass the computational and perceptual bottlenecks of pixel-level reconstruction, **Joint Embedding Predictive Architectures (JEPA)** predict future representations directly within abstract embedding spaces [2][3]. By avoiding pixel decoders, JEPA-based world models prevent model-averaging on ambiguous visual details and learn task-agnostic representations robust to visual distractors [2]. Furthermore, autoregressive sequence backbones and state-space models (such as S4WM) provide superior long-range memory and computational efficiency compared to standard recurrent networks [2][3].

## Generative World Simulators and Interactive Environments
A major milestone in generative artificial intelligence is the rise of video-generation models functioning as general-purpose world simulators [4][5]. Systems like OpenAI's Sora and Google DeepMind's Genie scale spatiotemporal (ST) transformers and diffusion architectures over compressed latent patches or discrete tokens [4][5][6]. Sora's spacetime patch tokenizer unifies video data across arbitrary resolutions, durations, and aspect ratios, supporting the generation of spatially and temporally consistent video rollouts [4][6].

Complementing text-to-video generation, Genie introduces unsupervised interactive environments by training large-scale models on unlabelled internet gameplay videos [5]. Utilizing a latent action model that infers discrete action tokens between video frames without supervision, Genie transforms passive video streams into interactive virtual worlds where users or agents can control characters on a frame-by-frame basis [5]. Despite these advances, generative simulators still face physical reasoning failures, such as occasional long-horizon drift and incorrect object state transformations [5][6].

## World Models in Large Language Models and Multimodal Foundation Models
Beyond embodied robotics and video generation, Large Language Models (LLMs) and Multimodal Foundation Models encode rich implicit world models through massive pretraining and next-token prediction [7][8][9]. Evaluated as text-based world models under interaction protocols, LLMs capture environment dynamics in partially observable Markov decision processes (POMDPs), scaling predictably with model capacity and yielding tangible downstream agent benefits (e.g., a 5.5% performance boost on WebShop via action verification and a 15% gain on SciWorld via warm-started reinforcement learning) [7][9].

Internal computation within these foundation models relies on distinct reasoning paradigms, including latent optimization (manipulating internal hidden states across multi-step inference), signal-guided control (utilizing planning tokens), and layer-recurrent execution [8][9]. When integrated with spatial reasoning benchmarks (such as CubeBench and WorldSimBench), multimodal world models bridge static semantic knowledge with dynamic physical simulation, supporting advanced agent planning and counterfactual reasoning [9].

## Robotics and Embodied AI Post-Training
The role of world models in robotics has shifted from auxiliary prediction mechanisms to active, interactive simulation environments for policy post-training and reinforcement learning [3][9]. Action-conditioned world models (e.g., UniSim, World-Gymnast, PlayWorld, WMPO) serve as surrogate environments where Vision-Language-Action (VLA) and diffusion policies roll out trajectories, receive intrinsic rewards, and optimize via reinforcement learning algorithms like Group Relative Policy Optimization (GRPO) [3][9]. 

To overcome distribution shift during policy optimization, modern architectures implement closed-loop co-evolution, alternating between real-world data collection to refine simulator dynamics and synthetic imagination data to improve embodied policy robustness [3].

## Trends and Open Problems
Despite rapid progress, several foundational challenges remain open in world model research [2][3][9]:
- **Compounding Error Drift:** In long-horizon imagination and planning, prediction errors compound exponentially, degrading policy performance. Mitigations such as residual-action modeling and structured Hamiltonian latents are active areas of study [2][3].
- **Sim-to-Real Transfer and Generalization:** Bridging the gap between synthetic simulator rollouts and complex real-world physical dynamics requires robust domain adaptation and self-correcting dynamics functions [3][9].
- **Scalable Physical Reasoning:** Enhancing multimodal world models to reliably simulate complex counterfactual physics (e.g., friction, fluid dynamics, and deformable object manipulation) remains an unsolved frontier [4][9].
- **Unified Multimodal Architectures:** Moving toward unified world models that seamlessly ingest text, video, audio, and tactile sensor streams to form a cohesive, universal simulator of reality [2][3].

## References
[1] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[2] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[3] World Model for Robot Learning: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2605.00080 (2026-05-01)
[4] Genie: Generative Interactive Environments. web. https://arxiv.org/abs/2402.15391 (2024-02-23)
[5] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[6] Can Large Language Models be Implicit Text-based World Models?. web. https://aclanthology.org/2026.acl-long.366.pdf (2026-01-01)
[7] Implicit Reasoning in Large Language Models: A Comprehensive Survey. arxiv. https://arxiv.org/html/2509.02350 (2025-09-02)
[8] Survey of World Models Based on Large Models. web. https://www.ecice06.com/EN/Y2026/V52/I6/1 (2026-06-15)
[9] World Models: A Comprehensive Survey of Knowledge, Architectures, and Applications. web. https://world-models.io/en/research/world-models-survey/ (2025-01-01)
[10] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. hf-search. https://huggingface.co/papers/2405.03520 (2024-05-06)
