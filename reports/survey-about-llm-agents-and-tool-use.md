# A Comprehensive Survey on Large Language Model Agents and Tool Use

## TL;DR
- **Cognitive Scaffolding & Planning**: Advanced reasoning frameworks—such as ReAct [1], Tree of Thoughts [2], and Reflexion [3]—interleave verbal reasoning with interactive actions, enabling agents to dynamically plan, search, and self-correct across multi-step execution horizons.
- **Tool Learning & API Integration**: Tool-augmented agents have evolved from static function matching to scaling across thousands of real-world REST APIs [4] and Model Context Protocol (MCP) servers [5], supported by structured parameter graphs [6] and self-evolution synthesis pipelines [7].
- **Multi-Agent Collaboration**: Multi-agent ecosystems (e.g., MultiAgentBench [8], AgentWebBench [9]) leverage graph communication topologies and OS-level container role separation to coordinate collaborative task solving and web navigation.
- **Security & Reliability Challenges**: Tool-using agents remain highly vulnerable to multi-turn indirect prompt injections (IPI) [10] and tool-selection hijacks [11], requiring rigorous inference-time causal diagnostics (AgentSentry [10], ARGUS [12]) and policy-first infrastructure gating [13].

## Background
Large Language Model (LLM) agents represent a paradigm shift from passive text generators to active problem-solvers capable of perceiving environments, formulating plans, and executing actions through external tools. Early agent systems relied on linear Chain-of-Thought (CoT) reasoning, which frequently suffered from error propagation and hallucinated outputs when handling complex, multi-step tasks. To bridge this gap, modern agent architectures synergize verbal reasoning with external tool interaction, bridging parametric memory with nonparametric operational environments.

Comprehending tool-using agents requires analyzing four interconnected pillars: (1) cognitive planning and reasoning frameworks that govern agent decision loops; (2) tool learning, API integration, and function-calling representations; (3) multi-agent collaboration protocols and benchmarking environments; and (4) security, alignment, and reliability safeguards against prompt injection and tool failure.

## Architecture, Planning, and Reasoning Frameworks
Agentic reasoning frameworks provide the cognitive scaffolding that dictates how an LLM interacts with external tools. Broadly classified into single-agent, tool-based, and multi-agent search methods [14], these frameworks replace brittle linear generation with iterative control loops.

The foundational paradigm is **ReAct** (Reasoning and Acting), which unifies verbal thought generation and interactive actions within an interleaved loop (`Thought` $\rightarrow$ `Act` $\rightarrow$ `Observation`) [1]. By grounding agent decisions in external environment observations (such as Wikipedia search APIs), ReAct overcomes the hallucination vulnerabilities inherent in static CoT. Extending beyond linear sequences, **Tree of Thoughts (ToT)** generalizes reasoning into a search tree, enabling LLMs to explore multiple reasoning branches, evaluate intermediate states via heuristic models, and perform Breadth-First or Depth-First backtracking from dead ends [2]. 

Furthermore, **Reflexion** introduces verbal reinforcement learning, allowing language agents to evaluate execution failures and store self-reflections in an episodic memory buffer [3]. This linguistic feedback loop adapts agent reasoning strategies across trials without requiring expensive model weight updates. Together, these reasoning structures govern the lifecycle of tool use—spanning autonomous tool selection, sequential chaining, parallel execution, and iterative refinement [14].

## Tool Learning, API Integration, and Function Calling
As the ecosystem of external tools expanded from isolated plugins to massive API repositories, tool learning emerged as a core subfield of agent research. Seminal frameworks such as **ToolLLM** established benchmarks over 16,464 real-world REST APIs using depth-first search-based decision trees (DFSDT) and neural API retrievers [4]. 

Recent advancements focus on structured parameter-level representations and data synthesis. **In-N-Out** converts API documentation into parameter-level graphs to facilitate multi-step dependency resolution [6], while **ToolACE** and **UniToolCall** leverage self-evolution synthesis and unified interaction modeling to enhance function-calling accuracy in resource-constrained models [7][15]. Moreover, the introduction of **DeepAgent** demonstrates how end-to-end deep reasoning combined with memory folding and reinforcement learning improves scalable toolset usage [16]. 

Despite these capabilities, evaluations highlight persistent challenges in parameter-value mapping, clarification requests, and unconstrained action tool usage [14]. An analysis of Model Context Protocol (MCP) servers reveals that action tools now constitute 65% of total tool downloads, with 94% of general-purpose server downloads involving unconstrained capabilities such as arbitrary code execution and browser automation [5].

## Multi-Agent Collaboration and Benchmarking
While single-agent architectures excel at bounded tasks, complex long-horizon workflows increasingly rely on Multi-Agent Systems (MAS) that distribute responsibilities across specialized roles (e.g., planners, executors, verifiers). 

Frameworks like **MultiAgentBench (MARBLE)** evaluate LLM agents across collaborative and competitive scenarios, demonstrating that decentralized graph-mesh communication topologies outperform star or hierarchical structures in open-ended research and synthesis tasks [8]. Similarly, **AgentWebBench** explores decentralized web information access, showing that agent coordination efficiency scales with model capacity and adaptive communication budgeting [9].

To rigorously evaluate these systems, benchmarks such as **AgentBench** and **TeamBench** have established standardized multi-dimensional evaluation suites [17]. Notably, TeamBench addresses the limitations of prompt-based role compliance by enforcing strict information asymmetry and role boundaries via OS-level container isolation [TeamBench]. Ablation studies reveal that LLM verifiers frequently experience false approvals (~49.4% approval of failed submissions), identifying validation as a primary bottleneck in multi-agent tool ecosystems [TeamBench].

## Security, Reliability, and Failure Modes
The integration of external tools exposes LLM agents to severe security vulnerabilities and operational failure modes. Chief among these is **Multi-Turn Indirect Prompt Injection (IPI)**, where untrusted data retrieved from web pages, documents, or tool outputs silently hijacks agent control flow.

Attack frameworks such as **ToolHijacker** demonstrate how optimized malicious tool descriptions can corrupt tool retrieval and selection phases, bypassing traditional prefix filters and alignment guardrails [11]. Similarly, **AgentBreaker** highlights that modern multi-agent web architectures remain susceptible to DOM-embedded prompt injections across planner-navigator pipelines [ACM:3832165].

To counteract temporal causal takeover, advanced inference-time defense mechanisms have been developed. **AgentSentry** employs boundary-anchored counterfactual re-execution and causally gated context purification to suppress attack-induced control signals while preserving task utility [10]. Concurrently, **ARGUS** utilizes influence-provenance graphs (IPG) and causal auditing to ensure that every state-changing action is supported by a benign causal chain [12]. At the infrastructure level, **Policy-First Tooling** and benchmark workbenches like **AgentCheck** provide model-agnostic permission layers and fault-injection testing to govern tool execution and prevent unsafe side effects [13][18].

## Trends and Open Problems
The literature reveals several prominent trends and persistent challenges in LLM agent research:
1. **From Function Calling to Agentic Workflows**: Transitioning from static one-shot API matching to autonomous, multi-turn tool discovery and execution via standardized protocols like MCP [5].
2. **Structural vs. Prompt-Based Control**: Moving beyond fragile prompt engineering toward OS-level container isolation [TeamBench], influence-provenance graphs [12], and policy-first infrastructure gating [13].
3. **Uncertainty and Environment Resilience**: Managing tool unreliability, malformed API outputs, and semantic ambiguities through robust error recovery and verification benchmarks [19][18].
4. **Causally Verified Alignment**: Addressing indirect prompt injection via inference-time causal diagnostics rather than superficial keyword filtering [10][12].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-06)
[2] Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arxiv. https://arxiv.org/abs/2305.10601 (2023-05-17)
[3] Reflexion: Language Agents with Verbal Reinforcement Learning. arxiv. https://arxiv.org/abs/2303.11366 (2023-03-20)
[4] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. web. https://arxiv.org/html/2307.16789v2 (2023-07-28)
[5] Monitoring Model Context Protocol (MCP) Servers and Action Tool Usage. web. https://arxiv.org/pdf/2603.23802 (2026-03-01)
[6] In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents. hf-search. https://huggingface.co/papers/2509.01560 (2025-12-30)
[7] ToolACE: Winning the Points of LLM Function Calling. hf-search. https://huggingface.co/papers/2409.00920 (2024-09-02)
[8] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents. hf-search. https://huggingface.co/papers/2503.01935 (2025-03-03)
[9] AgentWebBench: Benchmarking Multi-Agent Coordination in Agentic Web. arxiv. https://arxiv.org/abs/2604.10938v1 (2026-04-13)
[10] AgentSentry: Inference-Time Detection and Mitigation of Multi-Turn Indirect Prompt Injection in Tool-Augmented LLM Agents. arxiv. https://arxiv.org/pdf/2602.22724 (2026-02-22)
[11] Prompt Injection Attack to Tool Selection in LLM Agents. arxiv. https://arxiv.org/html/2504.19793v3 (2025-04-29)
[12] ARGUS: Defending LLM Agents Against Context-Aware Prompt Injection. arxiv. https://arxiv.org/pdf/2605.03378 (2026-07-08)
[13] Guardrails as Infrastructure: Policy-First Control for Tool-Orchestrated Workflows. hf-search. https://huggingface.co/papers/2603.18059 (2026-03-18)
[14] A Survey on Evaluation of LLM-based Agents. web. https://arxiv.org/html/2503.16416v2 (2025-03-24)
[15] UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents. hf-search. https://huggingface.co/papers/2604.11557 (2026-04-13)
[16] DeepAgent: A General Reasoning Agent with Scalable Toolsets. hf-search. https://huggingface.co/papers/2510.21618 (2025-10-24)
[17] AgentBench: Evaluating LLMs as Agents. hf-search. https://huggingface.co/papers/2308.03688 (2023-08-07)
[18] AgentCheck: A Reproduce-Intervene-Mitigate Workbench for LLM Agents over MCP. hf-search. https://huggingface.co/papers/2607.11098 (2026-07-14)
[19] Beyond Function Calling: Benchmarking Tool-Using Agents under Tool-Environment Unreliability. hf-search. https://huggingface.co/papers/2606.25819 (2026-06-24)
