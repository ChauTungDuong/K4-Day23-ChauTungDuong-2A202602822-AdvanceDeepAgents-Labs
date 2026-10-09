# Comprehensive Survey on Video and Multimodal Generation

## TL;DR
- Video and multimodal generation has transitioned from cascaded spatial-temporal U-Nets to scalable Diffusion Transformers (DiTs) and decoder-only autoregressive language models capable of modeling complex spacetime latent representations [1][2][3].
- Unified multimodal architectures (such as MM-LLMs and discrete tokenizers) enable flexible any-to-any synthesis across text, image, video, and Foley audio [4][5][6].
- Open-source ecosystems (e.g., Open-Sora, CogVideoX) and efficient hardware-software co-design quantization techniques (PARO, Q-VDiT) have democratized access to high-fidelity video generation by mitigating massive memory and compute bottlenecks [7][8][9][10].
- Rigorous hierarchical evaluation benchmark suites like VBench++ reveal persistent gaps in temporal consistency, motion dynamics, and cross-modal semantic alignment [11].

## Background
The rapid convergence of deep generative modeling and large-scale transformer architectures has revolutionized visual and multimodal synthesis. Historically, video generation relied heavily on 2D U-Net extensions with factorized temporal convolutions or cascaded super-resolution pipelines (e.g., Imagen Video [3]), which suffered from pipeline complexity, error accumulation, and limited temporal receptive fields. Inspired by the scaling laws of Large Language Models (LLMs), recent paradigms leverage transformer backbones operating on compressed spatial-temporal latent patches or unified discrete vocabularies. These advances bridge text, image, video, and audio into cohesive generative engines, paving the way toward interactive physical world simulators [1][2].

## Architectural Foundations of Video Generation
Modern video generation models predominantly adopt two architectural lineages: Diffusion Transformers (DiTs) and autoregressive decoder-only language models. Sora pioneered the spacetime latent patch representation, mapping compressed video latents into transformer tokens processed via self-attention across spatial and temporal dimensions without fixed-resolution constraints [1]. Similarly, Open-Sora implements an open-source Spatial-Temporal Diffusion Transformer (STDiT) utilizing 3D VAE compression and Rotary Positional Embeddings (RoPE) to support variable aspect ratios and durations [7]. CogVideoX introduces 3D full attention combined with an expert Transformer (AdaLN) to model spatial and temporal dimensions jointly, avoiding the instabilities of decoupled attention [8].

In parallel, autoregressive video generation has advanced through unified discrete sequence modeling. VideoPoet frames video generation as next-token prediction within a decoder-only LLM, uniting text, video, and audio into a shared discrete vocabulary of ~300,000 tokens using MAGVIT-v2 and SoundStream tokenizers [2]. This allows zero-shot task adaptation across text-to-video, image-to-video, and video editing. Cascaded diffusion architectures like Imagen Video demonstrated the power of deep scaling with 11.6B parameters across 7 sub-models but have largely been superseded by native single-stage or factorized diffusion transformers [3].

## Multimodal and Audio-Visual Generation
Beyond unimodal video synthesis, contemporary research emphasizes unified cross-modal understanding and generation (Any-to-Any models). NExT-GPT proposes a loosely-coupled three-tier MM-LLM integrating ImageBind encoders with a Vicuna-7B core and external latent diffusion decoders, governed by special modality signal tokens (`[IMGi]`, `[AUDi]`, `[VIDi]`) and Modality-Switching Instruction Tuning [4]. AnyGPT achieves true discrete sequence modeling within a standard LLaMA-2 7B backbone by decoupling semantic token generation from perceptual waveform/pixel de-tokenization [6].

For video-centric multimodal conditioning, Emu Video factorizes text-to-video generation into explicit image conditioning and video extrapolation, concatenating starting frame latents and binary masks channel-wise with input noise to eliminate semantic bottlenecks [12]. Furthermore, audio-visual synthesis has advanced with models like HunyuanVideo-Foley, which employs a hybrid MM-LLM/DiT architecture with Representation Alignment (REPA) loss to synchronize high-fidelity Foley audio with fine-grained video dynamics and text semantics [5].

## Datasets, Benchmarks, and Evaluation Protocols
Evaluating video and multimodal generation remains challenging due to the multi-dimensional nature of video quality (spatial realism, temporal flickering, physics adherence, and text-video alignment). Traditional metrics such as Fréchet Video Distance (FVD) and CLIP Score often fail to capture fine-grained perceptual flaws. VBench++ addresses this by providing a comprehensive, hierarchical benchmark suite that evaluates subject consistency, background consistency, motion smoothness, dynamic degree, and aesthetic quality [11]. These rigorous evaluations consistently highlight the performance trade-offs between static text-to-image realism and dynamic temporal consistency in video generators.

## Efficiency Optimizations: Quantization and Distillation
Scaling video diffusion transformers introduces severe computational bottlenecks, with 3D full-attention blocks consuming over 56 GB of memory per layer. To resolve this, PARO implements hardware-software co-design with pattern-aware reorder-based mixed-precision quantization, transforming diagonal attention patterns into block-diagonal structures to prevent outlier degradation in INT8/INT4 and delivering up to 2.71x speedups on NVIDIA A100 GPUs [9]. Concurrently, Q-VDiT introduces Token-aware Quantization Estimators (TQE) and Temporal Maintenance Distillation (TMD) to achieve state-of-the-art W3A6 quantization, preserving spatiotemporal correlations and motion dynamics without quality loss [10].

## Trends and Open Problems
Despite monumental progress, several active challenges and open problems persist in video and multimodal generation:
1. **Physical World Simulation and Causality**: While models like Sora capture intuitive physics, true adherence to Newtonian mechanics, object permanence, and long-horizon causal consistency remains imperfect [1].
2. **Computational and Memory Overhead**: High-resolution, long-duration generation (e.g., >10 seconds at 60fps) remains computationally prohibitive, demanding further breakthroughs in sub-quadratic attention, hardware-aware distillation, and efficient token compression [9][10].
3. **Data Quality and Copyright**: Sourcing massive, clean, high-definition video-audio-text paired datasets presents legal, ethical, and curation hurdles, propelling open-source curation initiatives like Open-Sora and CogVideoX [7][8].
4. **Unified Cross-Modal Fidelity**: Balancing semantic instruction-following with perceptual fidelity across text, audio, and video modalities requires more robust alignment techniques and unified tokenization strategies [4][5][6].

## References
[1] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[2] VideoPoet: A Large Language Model for Zero-Shot Video Generation. arxiv. https://arxiv.org/abs/2312.14125 (2023-12-21)
[3] Imagen Video: High Definition Video Generation with Diffusion Models. arxiv. https://arxiv.org/abs/2210.02303 (2022-10-05)
[4] NExT-GPT: Any-to-Any Multimodal LLM. hf-search. https://huggingface.co/papers/2309.05519 (2023-09-11)
[5] HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation. hf-search. https://huggingface.co/papers/2508.16930 (2025-08-23)
[6] AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling. hf-search. https://huggingface.co/papers/2402.12226 (2024-02-19)
[7] Open-Sora: Democratizing Efficient Video Production for All. hf-search. https://arxiv.org/abs/2412.20404 (2024-12-29)
[8] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. web. https://proceedings.iclr.cc/paper_files/paper/2025/file/ce31378e9f41d8907e97dab172b6c559-Paper-Conference.pdf (2025-05-01)
[9] PARO: Hardware-Software Co-design with Pattern aware Reorder-based Attention Quantization in Video Generation Models. web. https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/a4960209-c46c-4a78-aceb-1ea756a5fdac.pdf (2024-10-01)
[10] Q-VDiT: Towards Accurate Quantization and Distillation of Diffusion Transformers for Video Generation. web. https://openreview.net/forum?id=Oeo53q93iP (2025-06-18)
[11] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[12] Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. hf-search. https://huggingface.co/papers/2311.10709 (2023-11-17)
