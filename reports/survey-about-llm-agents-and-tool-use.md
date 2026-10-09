# LLM Agents and Tool Use: From Reason-Act Loops to Reliability Testing

## TL;DR

- ReAct supplies a foundational pattern: interleave language-model reasoning with actions that obtain external observations, then use those observations to revise the next step [1].
- Tool-use improvements target different bottlenecks: EasyTool condenses tool instructions, whereas ToolGym combines a large tool environment, planner-actor orchestration, and curated trajectories [2][3].
- One successful run is an incomplete reliability measure. ReliabilityBench tests consistency across runs, perturbations, and injected tool faults; its reported success falls under increasing perturbation [4].
- Security and evaluation are separate failure surfaces: Skill-Inject reports vulnerability to malicious skill-file instructions, while AgentProp-Bench flags unreliable judges and errors propagating from bad tool parameters [5][6].

## Background

A tool-using LLM agent is useful to think of as a loop rather than a single answer: it selects an action, receives an observation from an external source, and decides what to do next. This is a synthesis of the ReAct design, not a claim that every agent uses exactly that architecture [1]. ReAct interleaves reasoning traces and task-specific actions. Its authors argue that reasoning helps track and update plans and handle exceptions, while actions let the model consult knowledge bases or environments [1]. The distinction matters because an internally plausible answer need not be grounded in the current external state. On HotpotQA and FEVER, the ReAct abstract describes using a Wikipedia API to address hallucination and error propagation associated with chain-of-thought reasoning [1].

The foundational evidence also has boundaries. ReAct reports absolute success-rate improvements of 34% on ALFWorld and 10% on WebShop against the comparison methods described in its abstract, with one or two in-context examples [1]. Those results establish an advantage in the reported settings, not a general performance guarantee for arbitrary tools, users, or operating conditions. The retrieved material does not establish a direct comparison between ReAct and Toolformer; drawing one here would turn a plausible historical narrative into an unsupported finding [1]. This survey therefore compares documented mechanisms and reported evaluations, with special attention to the different questions that each experiment actually asks.

## The Interface Is Part of the Agent

Tool use depends not just on whether a model can reason, but on what information its interface presents. EasyTool proposes making tool instructions more concise and reports improved tool-use efficiency and performance in its retrieved summary [2]. The available evidence does not name the benchmarks or quantify the gains, so the defensible conclusion is narrower: concise documentation is a proposed intervention on the model's tool-facing context, not an established winner over alternative agent designs [2].

ReAct addresses a different interface boundary. Its actions obtain information from an external source; the returned observation can change the reasoning trace and subsequent plan [1]. Inference: documentation quality and observation quality can interact. A well-described tool may still return incomplete or failed results, while good results may be wasted if the agent chooses the wrong tool or arguments. This interaction is not directly measured by the EasyTool summary or the ReAct abstract [1][2]. It motivates evaluations that expose the model to imperfect tool states instead of treating a correct function call as equivalent to successful task completion.

## Orchestration and Training Data

ToolGym expands the scope from a single prompt technique to an environment for agent testing and data curation. Its abstract describes 5,571 format-unified tools across 204 apps, a task-creation engine for constrained long-horizon workflows, and a state controller that introduces interruptions and failures [3]. The proposed select-then-execute planner-actor framework separates reasoning and self-correction from execution [3]. Compared with ReAct's interleaving of reasoning and action within a trajectory, this is an architectural division of labor around planning and acting; the retrieved sources do not contain a controlled head-to-head comparison between the two [1][3].

ToolGym's authors report fine-tuning on 1,170 collected trajectories and superior performance to baselines using 119k samples [3]. This is a reported comparison, not proof that a specific data-efficiency ratio would persist under matched models, budgets, and tasks: those controls are not specified in the retrieved abstract [3]. Still, the result highlights a practical hypothesis. Trajectories that contain realistic constraints, tool failures, and correction opportunities might be more useful than a larger set of less targeted examples. That is an inference from the environment design and reported result, rather than a separately validated causal mechanism [3].

## What Counts as Success?

A final task outcome compresses a potentially fragile sequence of decisions into one number. ReliabilityBench explicitly challenges reliance on single-run success rates by measuring repeated-execution consistency with pass^k, robustness to semantically equivalent perturbations, and fault tolerance under controlled tool and API failures [4]. It uses end-state equivalence for correctness and injects timeouts, rate limits, partial responses, and schema drift [4]. This evaluation design distinguishes whether an agent can occasionally solve a task from whether it reliably reaches the intended state when conditions vary.

In its reported 1,280 episodes, ReliabilityBench observes success declining from 96.9% without perturbation to 88.1% at perturbation intensity 0.2; rate limiting is identified as the most damaging injected fault in its ablations [4]. Those figures refer to the authors' specific evaluation, which covers two models, two architectures, and four domains; they should not be presented as an industry-wide failure rate [4]. They nevertheless illustrate why a favorable baseline can conceal sensitivity to changes that do not alter the user's underlying request. An agent that succeeds once may not recover when a service responds late or changes its response shape [4].

AgentProp-Bench raises a complementary measurement problem. Its retrieved summary reports low reliability of automated judges, error propagation from bad tool parameters, and model-specific effects of runtime mitigation [6]. There are no retrieved agreement statistics or mitigation effect sizes, so the claim is about the categories of failure the authors report, not their prevalence across deployed agents [6]. Inference: evaluation should audit both the action trajectory and the grading procedure. Otherwise, a wrongly scored result or an early argument error can make final-answer comparisons difficult to interpret [6].

## Security at External-Content Boundaries

Tool use gives the agent access to more information, but also brings externally supplied content into its decision process. Skill-Inject investigates malicious instructions in skill files and its retrieved summary reports up to 80% vulnerability among frontier models [5]. The summary does not define the vulnerability metric or specify attack construction and tested models. It cannot establish real-world attack prevalence or a general success rate for prompt injection [5]. The narrower takeaway is that a workflow may complete benign tasks yet still be redirected by instructions encountered through an external artifact [5].

This is distinct from ReliabilityBench's service faults. A timeout or rate limit tests recovery when a tool fails to deliver normally; an adversarial skill file tests whether the agent preserves the boundary between external material and authorized instructions [4][5]. Both belong in a broader reliability program, but their measurements cannot be pooled into a single percentage. Similarly, AgentProp-Bench's bad-parameter cascades concern tool-call and evaluation behavior, not necessarily malicious input [6]. Treating these as separate mechanisms helps align defenses and tests with the actual failure mode [4][5][6].

## Trends and Open Problems

The retrieved 2024-2026 material shifts attention from tool calling as an isolated ability toward documentation, long-horizon environments, perturbation robustness, adversarial inputs, and trustworthy grading [2][3][4][5][6]. Inference: a useful next comparison would hold models and task budgets fixed while varying instruction compression, planning structure, trajectory training, and fault exposure. None of the retrieved sources by itself establishes which combination generalizes best [2][3][4].

Several limitations remain concrete. ToolGym's abstract does not provide enough baseline detail for a controlled data-efficiency conclusion; ReliabilityBench's limited model and domain coverage constrains generalization; the available Skill-Inject summary lacks a precise vulnerability definition; and the AgentProp-Bench summary omits judge-agreement statistics [3][4][5][6]. Evidence provenance is another limit: two supplied arXiv identifiers have no retrieved titles, abstracts, or findings in the research notes and are recorded only as unresolved records [7][8]. They are not evidence for claims about agent performance. A stronger survey would first retrieve and verify those records, then compare independently reproduced outcomes, full trajectories, and human-checked grading under matched conditions [3][4][6][7][8].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://huggingface.co/papers/2210.03629 (2022-10-06)
[2] EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction. hf-search. https://huggingface.co/papers/2401.06201 (2024-01-11)
[3] ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation. web. https://huggingface.co/papers/2601.06328 (2026-01-09)
[4] ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions. web. https://huggingface.co/papers/2601.06112 (2026-01-03)
[5] Skill-Inject: Measuring Agent Vulnerability to Skill File Attacks. hf-search. https://huggingface.co/papers/2602.20156 (2026-02-23)
[6] Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench. hf-search. https://huggingface.co/papers/2604.16706 (2026-04-17)
[7] unknown. arxiv. https://arxiv.org/abs/2609.35932 (unknown)
[8] unknown. arxiv. https://arxiv.org/abs/2609.36603 (unknown)
