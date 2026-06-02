# bayesian chip debug oracle!

you're dropped into a simulated chip that has a hidden fault. you run diagnostic tests for things like memory bandwidth, branch misprediction rate, cache hit rate... and after each one the app updates its probability estimate for every possible fault configuration using bayes' theorem. it then recommends which test to run next based on which one would reduce uncertainty the most (value of information).

the benchmark specs come from real hardware: intel core ultra 9 285k and amd ryzen 9 9950x datasheets, spec cpu2017 results, stream memory benchmarks, and intel vtune profiler docs. so the healthy vs degraded numbers are grounded in what these chips actually measure in the field!

the bayesian network structure where memory controller faults raise the probability of l2 cache faults is also based on actual thermal and electrical coupling between adjacent chip blocks (shared power rails, shared decode pipelines).

## features
- **cpu die floorplan** that heats up red as components get implicated
- **value of information chart** that reranks remaining tests after every update, showing which one is worth running next
- **posterior predictive** — a live mixture of two gaussians showing what the model expects to see before you run a test
- **bootstrap confidence intervals** on each component's fault probability (unlocks after 2 tests)
- **head-to-head mode** — entropy-guided vs random test ordering on the same hidden fault, side by side
- **geometric distribution fit** across 600 simulated trials, comparing how many tests each strategy needs to reach 90% confidence
- **chris-ometer** — prof. chris gregg reacts to the current system health in real time

## how to run

```
streamlit run app.py
```

enjoy!