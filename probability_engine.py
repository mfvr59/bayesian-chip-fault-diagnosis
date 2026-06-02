import numpy as np
import scipy.stats
from itertools import product
from typing import Dict, List, Tuple
from collections import Counter

COMPONENTS = [
    "Memory Controller",
    "L2/L3 Cache",
    "Out-of-Order Engine",
    "Branch Predictor",
    "L1 Cache",
    "PCIe Interface",
]
N = len(COMPONENTS)

BN_MARGINAL_PRIORS = {
    "Memory Controller": 0.18,
    "Out-of-Order Engine": 0.15,
    "PCIe Interface": 0.12,
}

BN_CONDITIONAL = {
    "L2/L3 Cache": {
        1: 0.30,
        0: 0.10,
    },
    "Branch Predictor": {
        1: 0.28,
        0: 0.07,
    },
    "L1 Cache": {
        1: 0.25,
        0: 0.08,
    },
}

BN_EDGES = [
    ("Memory Controller", "L2/L3 Cache"),
    ("Out-of-Order Engine", "Branch Predictor"),
    ("Out-of-Order Engine", "L1 Cache"),
]

# Returns the directed edges of the fault dependency graph.
def bayesian_network_edges():
    return BN_EDGES

# Returns every possible combination of component fault states as a list.
def enumerate_fault_states():
    return list(product([0, 1], repeat=N))

FAULT_STATES = enumerate_fault_states()

# Computes the prior probability for every fault state using the Bayesian network.
def compute_prior():
    mc_idx = COMPONENTS.index("Memory Controller")
    l2_idx = COMPONENTS.index("L2/L3 Cache")
    ooo_idx = COMPONENTS.index("Out-of-Order Engine")
    bp_idx = COMPONENTS.index("Branch Predictor")
    l1_idx = COMPONENTS.index("L1 Cache")
    pci_idx = COMPONENTS.index("PCIe Interface")

    p_mc = BN_MARGINAL_PRIORS["Memory Controller"]
    p_ooo = BN_MARGINAL_PRIORS["Out-of-Order Engine"]
    p_pci = BN_MARGINAL_PRIORS["PCIe Interface"]

    priors = {}
    for state in FAULT_STATES:
        p = 1.0
        if state[mc_idx] == 1:
            p = p * p_mc
        else:
            p = p * (1 - p_mc)
        if state[ooo_idx] == 1:
            p = p * p_ooo
        else:
            p = p * (1 - p_ooo)
        if state[pci_idx] == 1:
            p = p * p_pci
        else:
            p = p * (1 - p_pci)
        p_l2 = BN_CONDITIONAL["L2/L3 Cache"][state[mc_idx]]
        if state[l2_idx] == 1:
            p = p * p_l2
        else:
            p = p * (1 - p_l2)
        p_bp = BN_CONDITIONAL["Branch Predictor"][state[ooo_idx]]
        if state[bp_idx] == 1:
            p = p * p_bp
        else:
            p = p * (1 - p_bp)
        p_l1 = BN_CONDITIONAL["L1 Cache"][state[ooo_idx]]
        if state[l1_idx] == 1:
            p = p * p_l1
        else:
            p = p * (1 - p_l1)
        priors[state] = p

    total = 0.0
    for val in priors.values():
        total = total + val
    assert abs(total - 1.0) < 1e-9, f"Prior sums to {total}"
    return priors


TESTS = {
    "Branch Misprediction Rate": {
        "targets": ["Branch Predictor"],
        "mu_healthy": 0.04,
        "mu_degraded": 0.24,
        "sigma": 0.005,
    },
    "Instructions Per Cycle": {
        "targets": ["Out-of-Order Engine"],
        "mu_healthy": 3.8,
        "mu_degraded": 1.6,
        "sigma": 0.08,
    },
    "L3 Cache Hit Rate": {
        "targets": ["L2/L3 Cache"],
        "mu_healthy": 0.92,
        "mu_degraded": 0.58,
        "sigma": 0.015,
    },
    "Memory Bandwidth": {
        "targets": ["Memory Controller", "L2/L3 Cache"],
        "mu_healthy": 89.0,
        "mu_degraded": 38.0,
        "sigma": 1.5,
    },
    "PCIe Transfer Rate": {
        "targets": ["PCIe Interface"],
        "mu_healthy": 62.0,
        "mu_degraded": 15.0,
        "sigma": 0.5,
    },
    "Memory Latency": {
        "targets": ["Memory Controller"],
        "mu_healthy": 80.0,
        "mu_degraded": 210.0,
        "sigma": 3.0,
    },
    "L1 Cache Hit Rate": {
        "targets": ["L1 Cache"],
        "mu_healthy": 0.97,
        "mu_degraded": 0.72,
        "sigma": 0.012,
    },
}

TEST_NAMES = list(TESTS.keys())

# Returns the Gaussian likelihood of a measurement given a fault state.
def compute_likelihood(test_name, measurement, state):
    test = TESTS[test_name]
    targets = test["targets"]
    sigma = test["sigma"]
    any_broken = False
    for c in targets:
        if c in COMPONENTS:
            if state[COMPONENTS.index(c)] == 1:
                any_broken = True
    if any_broken:
        mu = test["mu_degraded"]
    else:
        mu = test["mu_healthy"]
    return scipy.stats.norm.pdf(measurement, loc=mu, scale=sigma)

# Updates the posterior distribution after observing a new measurement.
def bayesian_update(posterior, test_name, measurement):
    unnormalized = {}
    for s, p in posterior.items():
        unnormalized[s] = compute_likelihood(test_name, measurement, s) * p
    total = 0.0
    for val in unnormalized.values():
        total = total + val
    assert total > 0, "All likelihoods zero, measurement impossible under model"
    updated = {}
    for s, p in unnormalized.items():
        updated[s] = p / total
    total_check = 0.0
    for val in updated.values():
        total_check = total_check + val
    assert abs(total_check - 1.0) < 1e-9
    return updated

# Simulates a noisy measurement for a given test and true fault state.
def simulate_measurement(test_name, true_state):
    test = TESTS[test_name]
    any_broken = False
    for c in test["targets"]:
        if c in COMPONENTS:
            if true_state[COMPONENTS.index(c)] == 1:
                any_broken = True
    if any_broken:
        mu = test["mu_degraded"]
    else:
        mu = test["mu_healthy"]
    return np.random.normal(loc=mu, scale=test["sigma"])

# Computes the Shannon entropy of a probability distribution over fault states.
def entropy(posterior):
    h = 0.0
    for p in posterior.values():
        if p > 1e-300:
            h = h + p * np.log(p)
    return -h

# Estimates the expected entropy of the posterior after running a test.
def expected_posterior_entropy(posterior, test_name, n_samples=500):
    states = list(posterior.keys())
    probs = list(posterior.values())
    total_h = 0.0
    for _ in range(n_samples):
        hyp_state = states[np.random.choice(len(states), p=probs)]
        x = simulate_measurement(test_name, hyp_state)
        total_h = total_h + entropy(bayesian_update(posterior, test_name, x))
    return total_h / n_samples

# Returns how many bits of entropy we expect to eliminate by running a given test.
def value_of_information(posterior, test_name, n_samples=500):
    return entropy(posterior) - expected_posterior_entropy(posterior, test_name, n_samples)

# Returns the test with the highest value of information and all VoI scores.
def recommend_next_test(posterior, already_run=[], n_samples=500):
    remaining = []
    for t in TEST_NAMES:
        if t not in already_run:
            remaining.append(t)
    voi_scores = {}
    for t in remaining:
        voi_scores[t] = value_of_information(posterior, t, n_samples)
    best = max(voi_scores, key=voi_scores.get)
    return best, voi_scores

# Returns the marginal fault probability for each component by summing over all states.
def marginal_fault_probs(posterior):
    probs = {}
    for c in COMPONENTS:
        probs[c] = 0.0
    for state, prob in posterior.items():
        for i, broken in enumerate(state):
            if broken:
                probs[COMPONENTS[i]] = probs[COMPONENTS[i]] + prob
    return probs

# Returns the expected number of broken components using linearity of expectation.
def expected_fault_count(posterior):
    marg = marginal_fault_probs(posterior)
    total = 0.0
    for val in marg.values():
        total = total + val
    return total

# Returns the fault state with the highest posterior probability.
def map_fault_state(posterior):
    return max(posterior, key=posterior.get)

# Returns the probability of the single most likely fault state.
def posterior_confidence(posterior):
    return posterior[map_fault_state(posterior)]

# Samples a random fault state that has at least one broken component.
def _sample_faulty_state(prior):
    faulty = []
    for s in FAULT_STATES:
        if any(s):
            faulty.append(s)
    faulty_probs = []
    for s in faulty:
        faulty_probs.append(prior[s])
    faulty_probs = np.array(faulty_probs)
    faulty_probs = faulty_probs / faulty_probs.sum()
    return faulty[np.random.choice(len(faulty), p=faulty_probs)]

# Simulates many diagnostic sessions and fits a Geometric distribution to tests-to-diagnosis.
def geometric_tests_to_confidence(target_confidence=0.90, n_trials=600, strategies=("guided", "random"), n_samples_voi=60):
    prior = compute_prior()
    max_tests = len(TEST_NAMES)
    results = {}

    for strategy in strategies:
        t_values = []
        never = 0

        for _ in range(n_trials):
            true_state = _sample_faulty_state(prior)
            post = dict(prior)
            done = []
            reached = None

            for step in range(max_tests):
                remaining = []
                for t in TEST_NAMES:
                    if t not in done:
                        remaining.append(t)
                if not remaining:
                    break
                if strategy == "guided":
                    voi = {}
                    for t in remaining:
                        voi[t] = value_of_information(post, t, n_samples=n_samples_voi)
                    test = max(voi, key=voi.get)
                else:
                    test = np.random.choice(remaining)

                m = simulate_measurement(test, true_state)
                post = bayesian_update(post, test, m)
                done.append(test)

                if posterior_confidence(post) >= target_confidence:
                    reached = step + 1
                    break

            if reached is not None:
                t_values.append(reached)
            else:
                never = never + 1

        if t_values:
            mean_T = float(np.mean(t_values))
            std_T = float(np.std(t_values))
            p_hat = 1.0 / mean_T
        else:
            mean_T = None
            std_T = None
            p_hat = None

        support = list(range(1, max_tests + 1))
        pmf_vals = []
        if p_hat is not None:
            for k in support:
                pmf_vals.append((1 - p_hat) ** (k - 1) * p_hat)
        else:
            for k in support:
                pmf_vals.append(0.0)

        cnt = Counter(t_values)
        emp_pmf = []
        for k in support:
            emp_pmf.append(cnt.get(k, 0) / n_trials)

        results[strategy] = {
            "mean_T": mean_T,
            "std_T": std_T,
            "p_hat": p_hat,
            "pmf_support": support,
            "pmf_values": pmf_vals,
            "empirical_pmf": emp_pmf,
            "pct_never_reached": never / n_trials,
        }

    return results

# Computes 95% bootstrap confidence intervals on each component's marginal fault probability.
def bootstrap_marginal_cis(tests_run, prior, n_bootstrap=800, confidence=0.95):
    if not tests_run:
        result = {}
        for c in COMPONENTS:
            result[c] = (0.0, 1.0)
        return result

    n = len(tests_run)
    boot_samples = {}
    for c in COMPONENTS:
        boot_samples[c] = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_post = dict(prior)
        for idx in indices:
            test_name, measurement = tests_run[idx]
            boot_post = bayesian_update(boot_post, test_name, measurement)
        marg = marginal_fault_probs(boot_post)
        for c in COMPONENTS:
            boot_samples[c][b] = marg[c]

    alpha = 1 - confidence
    cis = {}
    for c in COMPONENTS:
        sv = np.sort(boot_samples[c])
        lo = float(sv[int(alpha / 2 * n_bootstrap)])
        hi = float(sv[int((1 - alpha / 2) * n_bootstrap)])
        cis[c] = (lo, hi)
    return cis

# Computes the posterior predictive distribution as a mixture of two Gaussians.
def posterior_predictive_pdf(test_name, posterior, n_points=300):
    test = TESTS[test_name]
    mu_h = test["mu_healthy"]
    mu_d = test["mu_degraded"]
    sigma = test["sigma"]
    targets = test["targets"]

    p_deg = 0.0
    for state, p in posterior.items():
        for c in targets:
            if c in COMPONENTS:
                if state[COMPONENTS.index(c)] == 1:
                    p_deg = p_deg + p
                    break
    p_hlt = 1.0 - p_deg

    x = np.linspace(min(mu_h, mu_d) - 4.5 * sigma, max(mu_h, mu_d) + 4.5 * sigma, n_points)
    pdf = p_hlt * scipy.stats.norm.pdf(x, mu_h, sigma) + p_deg * scipy.stats.norm.pdf(x, mu_d, sigma)
    return x, pdf, float(p_deg), float(p_hlt)

# Runs a Bayesian hypothesis test per component and returns whether each is likely broken.
def component_hypothesis_tests(posterior, alpha=0.10):
    marginals = marginal_fault_probs(posterior)
    results = {}
    for comp in COMPONENTS:
        fault_prob = marginals[comp]
        p_value = 1.0 - fault_prob
        if p_value < 0.01:
            evidence = "very strong"
        elif p_value < 0.05:
            evidence = "strong"
        elif p_value < alpha:
            evidence = "moderate"
        elif p_value < 0.20:
            evidence = "weak"
        else:
            evidence = "none"
        results[comp] = {
            "fault_prob": fault_prob,
            "p_value": p_value,
            "reject": p_value < alpha,
            "evidence": evidence,
        }
    return results

def run_validation():
    print("=" * 60)
    print("VALIDATION SUITE — CPU Bayesian Fault Diagnosis Engine")
    print("=" * 60)

    prior = compute_prior()

    total = 0.0
    for val in prior.values():
        total = total + val
    assert abs(total - 1.0) < 1e-9, f"Prior sums to {total}"
    print(f"✓ Prior sums to 1.0 (got {total:.10f})")

    marg = marginal_fault_probs(prior)
    expected_l2 = (BN_CONDITIONAL["L2/L3 Cache"][1] * BN_MARGINAL_PRIORS["Memory Controller"]
                   + BN_CONDITIONAL["L2/L3 Cache"][0] * (1 - BN_MARGINAL_PRIORS["Memory Controller"]))
    assert abs(marg["L2/L3 Cache"] - expected_l2) < 1e-9
    print(f"✓ BN marginals consistent: P(L2 broken)={marg['L2/L3 Cache']:.4f} == {expected_l2:.4f}")

    mc_idx = COMPONENTS.index("Memory Controller")
    l2_idx = COMPONENTS.index("L2/L3 Cache")
    mc_broken = {}
    mc_healthy = {}
    for s, p in prior.items():
        if s[mc_idx] == 1:
            mc_broken[s] = p
        else:
            mc_healthy[s] = p
    z_broken = 0.0
    for val in mc_broken.values():
        z_broken = z_broken + val
    z_healthy = 0.0
    for val in mc_healthy.values():
        z_healthy = z_healthy + val
    p_l2_mc1 = 0.0
    for s, p in mc_broken.items():
        if s[l2_idx] == 1:
            p_l2_mc1 = p_l2_mc1 + p
    p_l2_mc1 = p_l2_mc1 / z_broken
    p_l2_mc0 = 0.0
    for s, p in mc_healthy.items():
        if s[l2_idx] == 1:
            p_l2_mc0 = p_l2_mc0 + p
    p_l2_mc0 = p_l2_mc0 / z_healthy
    assert p_l2_mc1 > p_l2_mc0
    print(f"✓ BN correlation: P(L2|MC broken)={p_l2_mc1:.3f} > P(L2|MC healthy)={p_l2_mc0:.3f}")

    orig_sigma = TESTS["Memory Bandwidth"]["sigma"]
    TESTS["Memory Bandwidth"]["sigma"] = 0.001
    updated = bayesian_update(prior, "Memory Bandwidth", TESTS["Memory Bandwidth"]["mu_degraded"])
    for state, prob in updated.items():
        if state[mc_idx] == 0 and state[l2_idx] == 0:
            assert prob < 1e-5, f"State {state} should be ~0, got {prob}"
    print("✓ Near-noiseless test collapses posterior correctly")
    TESTS["Memory Bandwidth"]["sigma"] = orig_sigma

    np.random.seed(42)
    true_state = (1, 1, 0, 0, 0, 0)
    post = dict(prior)
    h0 = entropy(prior)
    for t in ["Memory Bandwidth", "Memory Latency", "L3 Cache Hit Rate"]:
        m = simulate_measurement(t, true_state)
        post = bayesian_update(post, t, m)
    assert entropy(post) < h0
    print(f"✓ Entropy decreased: {h0:.3f} → {entropy(post):.3f}")

    updated_bw = bayesian_update(prior, "Memory Bandwidth", TESTS["Memory Bandwidth"]["mu_degraded"])
    marg_updated = marginal_fault_probs(updated_bw)
    assert marg_updated["L2/L3 Cache"] > marg_updated["Branch Predictor"]
    print(f"✓ Targeted update: P(L2 broken)={marg_updated['L2/L3 Cache']:.3f} "
          f"> P(BP broken)={marg_updated['Branch Predictor']:.3f} after memory bandwidth test")

    print("\n✓ ALL VALIDATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
    print()

    np.random.seed(7)
    prior = compute_prior()

    true_state = (1, 1, 0, 0, 0, 0)
    true_faults = []
    for i, b in enumerate(true_state):
        if b:
            true_faults.append(COMPONENTS[i])
    print(f"True faults (hidden): {true_faults}\n")

    post = dict(prior)
    done = []

    for step in range(6):
        best, voi_scores = recommend_next_test(post, done, n_samples=200)
        m = simulate_measurement(best, true_state)
        post = bayesian_update(post, best, m)
        done.append(best)
        marg = marginal_fault_probs(post)
        print(f"Test {step+1}: {best}")
        print(f"  Measurement: {m:.3f}  |  H={entropy(post):.3f}  |  Conf={posterior_confidence(post):.1%}")
        for comp, prob in marg.items():
            broken_indices = []
            for i, b in enumerate(true_state):
                if b:
                    broken_indices.append(i)
            if COMPONENTS.index(comp) in broken_indices:
                flag = " ← BROKEN"
            else:
                flag = ""
            print(f"    {comp:25s}: {prob:.3f} {'█'*int(prob*20)}{flag}")
        print()

    ms = map_fault_state(post)
    map_faults = []
    for i, b in enumerate(ms):
        if b:
            map_faults.append(COMPONENTS[i])
    print(f"MAP diagnosis: {map_faults}")
    print(f"True faults:   {true_faults}")
    print(f"Correct:       {ms == true_state}")
