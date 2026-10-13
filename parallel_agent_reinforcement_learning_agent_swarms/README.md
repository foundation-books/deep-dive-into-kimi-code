# Chapter 26 deterministic fixtures

This directory makes the chapter-created graph, work/critical-step, reward,
context, aggregation, and handoff examples executable.  It is intentionally
dependency-free and uses exact arithmetic where ratios appear.

Run the checks with:

```bash
cd code
python3 -B -m unittest \
  parallel_agent_reinforcement_learning_agent_swarms.test_parallel_agent_reinforcement_learning_agent_swarms_reference
```

The ten `CH25-D*` checks validate teaching semantics only.  They do not launch
workers, train PARL, reproduce Kimi Agent Swarm, or estimate benchmark effects.
