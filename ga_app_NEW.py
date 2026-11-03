# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters (updated as requested) ----
POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
N_GENERATIONS = 50

# ---- GA Hyperparameters (simple & sensible) ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN  # ~1 bit flip expected per chromosome

ELITISM = 2

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness reaches its maximum value of 80
    when the number of ones equals 50.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)  # cut between [1 .. CHROM_LEN-1]
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(individual: np.ndarray) -> np.ndarray:
    individual = individual.copy()
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(pop: np.ndarray, generations: int):
    best_fitness_per_gen = []
    best_individual = None
    best_f = -np.inf

    for gen in range(generations + 1):

        # Evaluate current population
        fits = np.array([fitness(ind) for ind in pop])

        # Find best individual in current generation
        gen_best_idx = np.argmax(fits)
        gen_best = pop[gen_best_idx]
        gen_best_f = fits[gen_best_idx]

        best_fitness_per_gen.append(float(gen_best_f))

        # Update overall best solution
        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = gen_best.copy()

        # Stop after evaluating generation 50
        if gen == generations:
            break

        # Elitism: preserve the best individuals
        elite_idx = np.argsort(fits)[-ELITISM:]
        elites = pop[elite_idx].copy()

        # Generate next population
        new_pop = []

        while len(new_pop) < len(pop) - ELITISM:

            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)

            c1, c2 = single_point_crossover(p1, p2)

            c1 = mutate(c1)
            c2 = mutate(c2)

            new_pop.extend([c1, c2])

        # Add preserved elite individuals
        pop = np.array(
            new_pop[:len(pop) - ELITISM] + list(elites),
            dtype=np.int8
        )

    return best_individual, best_f, best_fitness_per_gen

# ---- Streamlit UI ----
st.set_page_config(
    page_title="Genetic Algorithm (80-bit Pattern)",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Algorithm: Evolve a 80-bit Pattern Optimization")
st.caption(
    "Fixed requirements: Population = 300, "
    "Chromosome Length = 80, Generations = 50. "
    "Maximum fitness = 80 when the number of ones equals 50."
)

with st.expander("ℹ️ Problem setup (fixed by requirement)", expanded=True):
    st.write(
        f"""
- **Population size**: `{POP_SIZE}`  
- **Chromosome length**: `{CHROM_LEN}`  
- **Target number of ones**: `{TARGET_ONES}`  
- **Max fitness at optimum**: `{MAX_FITNESS}` (when ones = {TARGET_ONES})  
- **Generations**: `{N_GENERATIONS}`  
- **Selection**: Tournament (k={TOURNAMENT_K})  
- **Crossover**: Single-point (rate={CROSSOVER_RATE})  
- **Mutation**: Bit-flip (per-bit rate={MUTATION_RATE:.4f})
"""
    )

col1, col2 = st.columns(2)
with col1:
    seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)
with col2:
    run_btn = st.button("Run Genetic Algorithm", type="primary")

if run_btn:
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialize and evolve
    population = init_population(POP_SIZE, CHROM_LEN)
    best_ind, best_fit, curve = evolve(population, N_GENERATIONS)

    # Results
    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Best Individual Found")
    st.metric(label="Best Fitness", value=f"{best_fit:.0f}")
    st.write(f"**Ones**: {ones_count} | **Zeros**: {zeros_count} | **Length**: {CHROM_LEN}")
    st.code(bitstring, language="text")

    # Convergence plot
    st.subheader("📉 Convergence (Best Fitness per Generation)")
    fig, ax = plt.subplots()
    ax.plot(range(len(curve)), curve, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("GA Convergence")
    ax.set_xticks(range(0, N_GENERATIONS + 1, 5))
    ax.grid(True, linestyle="--", alpha=0.5)

    # Validation note
    if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
        st.success("Perfect match achieved: 50 ones maximum fitness = 80!")
    else:
        st.info("GA may reach near-optimal solutions; try another seed to explore further.")

st.caption( 
    "Genetic Algorithm Optimization | "
    "80-bit chromosome | Target: 50 ones"
)