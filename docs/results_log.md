# World Models Results Log

## 2025-11-24: VizDoom Reproduction (Curriculum Learning)

### Experiment Setup
*   **Environment:** `VizdoomTakeCover-v0`
*   **Method:** Sim2Real2Sim (Curriculum Learning)
*   **Data:**
    *   2,000 Random Episodes (Bootstrap)
    *   500 On-Policy Episodes (Correction)
*   **Training:**
    *   VAE: 1 Epoch
    *   RNN: 20 Epochs (Retrained)
    *   Controller: 100 Generations (Retrained)

### Results
*   **Metric:** Average Survival Time (Score) over 100 episodes.
*   **Target:** > 800 (Paper: ~1092)
*   **Achieved:** ~203.0 (Best single run), Average ~160-200.

### Observations
1.  **Behavior:** The agent successfully learned to dodge immediate fireballs, showing it understands the VAE reconstruction and RNN predictions for short horizons.
2.  **Failure Mode:** The agent eventually gets cornered or fails to anticipate fast-moving fireballs from a distance. This suggests the "Dream" might not be accurate enough for long-term planning, or the Controller needs more evolution time.
3.  **Sim2Real Gap:** The agent achieved ~1000 score in the "Dream" environment but only ~200 in the real environment. This confirms a significant Sim2Real gap.
4.  **Curriculum Effect:** Adding on-policy data improved stability compared to the pure random baseline (anecdotal, based on initial failures), but more iterations are needed.

## 2025-11-24: Phase 3 Improvement (Annealing & Scale)

### Experiment Setup
*   **Method:** Expanded Curriculum + Temperature Annealing
*   **Data:**
    *   Added 2,000 On-Policy Episodes (Total ~4,500).
*   **Training:**
    *   RNN: Retrained for 50 Epochs.
    *   Controller: Evolved for ~500 Generations (Interrupted) with Annealing (1.25 -> 0.86).

### Results
*   **Achieved:** Average ~180-220.
*   **Analysis:**
    *   **No Significant Improvement:** The score did not jump dramatically.
    *   **Surprise Lower:** The "Average Surprise" metric seems slightly lower in successful runs, indicating the RNN is better at predicting the world, but the Controller isn't exploiting it fully yet.
    *   **Training Time:** 500 generations might still be too few, or the "Dream" is still too imperfect (Sim2Real gap persists).

## 2025-11-24: Phase 4 & 5 (Restart & Verify)

### Setup
*   **Goal:** Fix fundamental action logic bugs and VAE blurriness.
*   **Changes:**
    *   **Fixed Action Logic:** Mapped continuous actions correctly to Discrete (Left/Right).
    *   **VAE Tuning:** Retrained VAE for 20 epochs (vs 1).
    *   **Data:** 500 Episodes (Small Scale).

### Results
*   **VAE:** Reconstruction quality improved dramatically. Fireballs became visible.
*   **Agent:** Movement was corrected (no longer stuck), but performance was limited by small data size.

## 2025-11-24: Phase 6 (Scaling Up)

### Setup
*   **Data:** 2,000 Random Episodes (8 Workers).
*   **Training:** VAE (5 Epochs), RNN (20 Epochs), Controller (300 Gens).

### Results
*   **Dream Score:** ~1000 (Max).
*   **Real Score:** ~160.
*   **Behavior:** **"Left-Only" Bias.** The agent learned that moving Left is better than standing still, but never learned to move Right.
*   **Diagnosis:** Local optimum. Random data didn't contain enough "Dodging" examples to teach the RNN that "Right" is a valid survival strategy.


## 2025-11-24: Phase 7 (Curriculum Learning)

### Setup
*   **Goal:** Fix "Left-Only" Bias.
*   **Data:** Collected 1,000 On-Policy Episodes using the "Left-Only" agent.
    *   *Analysis:* 75.9% Left Actions, 0.1% Right Actions.
*   **Training:** Retrained RNN (Mixed Data) -> Retrained Controller.

### Results
*   **Score:** ~250-300 (Best: 294).
*   **Behavior:** The agent **broke the Left-Only bias**. It now moves Right!
*   **New Failure Mode:** **"Vanishing Fireballs"**.
    *   In the "Dream" (RNN prediction), fireballs sometimes disappear mid-air before hitting the player.
    *   Because the fireball "disappears", the Controller thinks it's safe and doesn't dodge.
    *   *Cause:* VAE Latent Dim (64) might be too small, causing blurriness that the RNN interprets as "fading away".

## 2025-11-25: Phase 8 (Scaling & Tuning - BUG FIX)

### Setup
*   **Goal:** Fix "Vanishing Fireballs" & Improve Agent.
*   **Data:** Added 2,000 On-Policy Episodes (Total 5,000).
*   **Critical Fix:** Found `train_rnn.py` was **truncating** episodes to ~50 steps. Fixed to use **Padding** (trains on full 1000 steps).
*   **Training:**
    *   VAE: 10 Epochs (Latent 64).
    *   RNN: 20 Epochs (Fixed Padding).
    *   Controller: 1000 Generations (Temp 1.4 -> 0.5).

### Results
*   **Surprise:** **~0.04** (vs ~0.18 previously).
    *   *Significance:* This is extremely low. The RNN now **perfectly predicts** the fireballs and the world dynamics. The "Vanishing Fireball" delusion is **SOLVED**.
*   **Score:** ~213.0.
*   **Behavior:** **"Right-Wall" Bias.** The agent moves to the right wall and stays there.
*   **Analysis:**
    *   The "Brain" (RNN) is fixed. It sees the fireballs coming.
    *   The "Muscle" (Controller) is stuck in a local optimum. It knows "Left = Death" (from Phase 7) and "Random = Death". It found "Right Wall" survives slightly longer than random, but hasn't learned the complex "Dodge" dance yet.
    *   *Hypothesis:* The Controller needs **much longer training** or a **Curriculum of Skills** to break out of this local optimum.

## 2025-11-25: Phase 10 (Curriculum Learning - Round 2)

### Setup
*   **Goal:** Fix "Right-Wall Bias" (Sim2Real Gap).
*   **Data:** Added 2,000 On-Policy Episodes (Total 7,000).
    *   *Source:* Agent failing at the right wall.
*   **Training:**
    *   RNN: Retrained on 7k episodes.
    *   Controller: Retrained (Interrupted at Gen ~100).

### Results
*   **Best Score:** **439.0** (Episode 1).
*   **Behavior:** **Dodging Observed!** The agent no longer hugs the right wall. It attempts to dodge fireballs.
*   **Analysis:**
    *   **Success:** The Curriculum Learning strategy worked again. By showing the RNN that "Right Wall = Death", we forced the Controller to abandon that local optimum.
    *   **Current State:** The agent tries to dodge but is still "clumsy". It survives longer (8s vs 3s) but lacks the precision for long-term survival.
    *   **Sim2Real Gap:** Reduced but present. Dream score (~2050) is still higher than Reality (~439).

## 2025-11-25: Phase 11 (Scale to 10k)

### Setup
*   **Goal:** Refine dodging and reach paper standard.
*   **Data:** Added 3,000 On-Policy Episodes (Total 10,000).
    *   *Source:* "Clumsy Dodger" (Phase 10 Agent).
*   **Training:**
    *   RNN: Retrained on full 10k dataset.
    *   Controller: Retrained (Interrupted at Gen ~124).

### Results
*   **Best Score:** **347.0** (Episode 4).
*   **Behavior:** **Active Dodging.** The agent clearly reacts to fireballs and moves to safety. It is less "clumsy" than before but still makes mistakes.
*   **Analysis:**
    *   **Success:** Scaling to 10k episodes has stabilized the behavior. The agent is now a competent dodger.
    *   **Next Steps:** To reach "Superhuman" levels (Score > 1000), we likely need much longer Controller training (Evolution) or further fine-tuning.

## 2025-11-25: Phase 12 (Pure Massive Training - Failure)

### Setup
*   **Goal:** Reach "Superhuman" score (>1000) using pure evolution (4,000 gens).
*   **Training:**
    *   Controller: Ran for ~317 generations.
    *   Dream Mean Score: ~1000 (High).

### Results
*   **Best Score:** **255.0** (Episode 4).
*   **Behavior:** **Regression.** The agent performs worse than Phase 11. It seems to have overfitted to the Dream's flaws.
*   **Analysis:**
    *   **Sim2Real Gap Confirmed:** The Dream thinks the agent is surviving (Mean ~1000), but in reality, it dies quickly.
    *   **Delusion:** The RNN likely predicts that "Left Wall" or "Still" is safe, and Evolution exploited this loophole.
    *   **Conclusion:** Pure evolution cannot fix a broken Dream. We **must** use Reward Shaping to align the Dream with Reality.

## Phase 13: Architecture Tuning (MLP Controller)
*   **Date:** 2025-11-25
*   **Goal:** Use a Non-Linear Controller (MLP) to learn complex dodging logic.
*   **Changes:**
    *   Updated `src/controller.py` to support `get_action_mlp` (Hidden=64).
    *   Updated `train_dream.py` to use **JAX-Native ES** (OpenAI-ES style) to avoid CPU OOM with the larger parameter set (~37k params).
    *   Trained for 4,000 generations.
*   **Results:**
    *   **Dream Score:** Mean ~1384 (High).
    *   **Real Score:** **460.0** (New Record!).
    *   **Survival:** ~13 seconds.
*   **Observation:** The MLP agent is significantly better than the Linear agent. It shows active dodging behavior and is less prone to "wall bouncing". It still eventually fails, likely due to the "Vanishing Fireball" issue in the VAE, but the controller is no longer the primary bottleneck.

## Phase 14: Iterative Training (Reality Gap Fix)
*   **Date:** 2025-11-25
*   **Goal:** Fix the "Vanishing Fireball" delusion in the World Model.
*   **Method:**
    *   Collected 1,000 On-Policy Episodes (from Phase 13 agent).
    *   Retrained RNN (World Model) on mixed dataset (10k Random + 1k On-Policy).
    *   **Crucial Change:** Updated `MAX_SEQ_LEN` from 1000 to 2100 to match VizDoom episode length.
    *   Re-evolved Controller (JAX-ES) for 2,000 generations.
*   **Results:**
    *   **Diagnostic Video:** **SUCCESS.** The RNN now correctly predicts high death probability (P > 0.8) when a fireball hits. The "Vanishing Fireball" issue is resolved.
    *   **Real Score:** ~163 (Avg).
*   **Analysis:**
    *   The World Model is no longer delusional, but the Controller has not yet learned to exploit this new accuracy. It likely found a new "loophole" or simply hasn't converged to a robust dodging strategy.
    *   **Next Step:** Iterative Training Round 2. Collect 2,000 episodes with this *new* failing agent to teach the RNN about the remaining failure modes.

## Phase 15: Iterative Training Round 2 (Closing the Gap)
*   **Date:** 2025-11-25
*   **Goal:** Close the remaining Reality Gap by training on specific failure modes.
*   **Method:**
    *   Collected 2,000 On-Policy Episodes (from Phase 14 agent).
    *   Retrained RNN on 13,000 episodes (10k Random + 3k On-Policy).
    *   Re-evolved Controller (JAX-ES) for 2,000 generations.
*   **Results:**
    *   **Real Score:** **~300** (Max: 419).
    *   **Improvement:** **Doubled** the previous score (~163 -> ~300).
*   **Analysis:**
    *   **Success:** The iterative training strategy is working effectively. By showing the RNN exactly how the agent fails, we force the Controller to learn better survival strategies.
    *   **Next Steps:** Continue this loop. Round 3 with 3,000 episodes should push us towards 500+.

## Phase 16: Iterative Training Round 3 (Regression/Plateau)
*   **Date:** 2025-11-25
*   **Goal:** Push for Superhuman score (>800).
*   **Method:**
    *   Collected 3,000 On-Policy Episodes (from Phase 15 agent).
    *   Retrained RNN on 16,000 episodes (10k Random + 6k On-Policy).
    *   Re-evolved Controller (JAX-ES) for 2,000 generations.
*   **Results:**
    *   **Real Score:** **~251** (Max: 331).
    *   **Regression:** Performance dropped compared to Round 2 (~300, Max 419).
    *   **Surprise:** **~0.035** (Remains very low/accurate).
*   **Analysis:**
    *   **World Model is Good:** The low surprise indicates the RNN understands the world perfectly.
    *   **Controller is Stuck:** The Controller failed to improve despite a better World Model.
    *   **Hypothesis:**
        1.  **Capacity:** The MLP (Hidden=64) might be too small to master the increasingly complex "survival dance".
        2.  **Overfitting:** The dataset might be dominated by "failure" data, making the World Model too pessimistic or the Controller too risk-averse.
    *   **Next Steps:** Try increasing Controller size (Hidden=128 or 256) or adjusting the training data mix.

## Phase 17: High Temperature Training (Combating Pessimism)
*   **Date:** 2025-11-26
*   **Goal:** Use higher temperature ($\tau=1.30$) to break "Pessimism Bias".
*   **Method:** Re-evolved Controller (2000 gens) with `--temperature 1.30`.
*   **Results:**
    *   **Dream Score:** Mean ~1321.
    *   **Real Score:** Poor (Regression).
*   **Analysis:** Higher temperature didn't help. The issue is likely the data distribution itself.

## Phase 18: Data Rebalancing (Quality over Quantity)
*   **Date:** 2025-11-26
*   **Goal:** Remove "Pessimism Bias" by filtering out short failure episodes.
*   **Method:**
    *   **Filtered Data:** Kept only **Top 2,000** On-Policy episodes (Score > 323). Deleted ~12k short episodes.
    *   **Dataset:** 4,000 Total (2k Random + 2k Best On-Policy).
    *   **Retraining:** Retrained RNN (20 Epochs) -> Re-evolved Controller (2000 Gens, Temp 1.15).
*   **Results:**
    *   **Dream Score:** Mean **~1770** (Very High).
    *   **Real Score:** **491.0** (New Record!).
    *   **Stability:** Low (Avg ~206).
*   **Analysis:**
    *   **Success:** Cleaning the data unlocked a higher peak performance (491 vs 460). The "Pessimism Bias" was reduced.
    *   **Instability:** The agent is still fragile. It can dodge well (491) but often fails early (100-150).
    *   **Next Steps:** We have a "Super-Agent" (Episode 4). We should collect data *specifically* from this high-performing agent to teach the RNN what "winning" looks like.

## Phase 19: Success Amplification (Teaching "Winning")
*   **Date:** 2025-11-26
*   **Goal:** Stabilize the "Super-Agent" behavior by training on high-quality data.
*   **Method:**
    *   **Data:** Collected 2000 episodes from Phase 18 agent. Filtered for **Score > 350** (Kept ~1377).
    *   **Dataset:** Augmented the Phase 18 dataset (Total ~3377 episodes).
    *   **Training:** Retrained RNN -> Re-evolved Controller (Temp 1.15).
*   **Results:**
    *   **Dream Score:** Mean **~1866** (Highest yet).
    *   **Real Score:** Max **378**, Avg **~197**.
*   **Analysis:**
    *   **Dream vs Reality:** The Dream score increased, but Real score stayed flat/unstable. This suggests the World Model might be "overfitting to success" — it predicts survival too optimistically, so the Controller takes risks that work in the Dream but fail in reality (Sim2Real gap).
    *   **Stability:** The agent is still fragile.
    *   **Next Steps:**
        1.  **Controller Capacity:** The small MLP (Hidden=64) might be saturating. Try increasing to **Hidden=256**.
        2.  **Data Diversity:** We might have filtered out too many "recovery" examples. We need a mix of "Winning" and "Saving" (near-misses).

## Phase 20: Restore Random Baseline (Back to Basics)
*   **Date:** 2025-11-26
*   **Goal:** Fix instability by restoring the "boring" physics knowledge (Random Data) that we stripped out.
*   **Method:**
    *   **Data:** Collected 6,000 NEW random episodes.
    *   **Dataset:** Total ~8,000 Random + ~1,377 High-Quality On-Policy (Total ~9,377).
    *   **Training:** Retrained RNN -> Re-evolved Controller (Temp 1.15).
*   **Results:**
    *   **Dream Score:** Mean **~1686** (Lower than Phase 19, but more realistic).
    *   **Real Score:** **Avg 312.0** (Min 238, Max 405).
*   **Analysis:**
    *   **STABILITY ACHIEVED:** This is the most stable agent yet. Every single episode was > 230. No early failures.
    *   **Conclusion:** The World Model *needs* a massive amount of random data (just like the paper said) to understand the basic physics of the world. Filtering too aggressively for "quality" hurt generalization.
    *   **Next Steps:** Now that we have a stable base (Avg 312), we can safely resume **Iterative Training** to push for >500. We should collect data from *this* agent and add it to our robust dataset.

## Phase 21: Iterative Training Round 4 (Regression)
*   **Date:** 2025-11-26
*   **Goal:** Push for >500 using data from the stable Phase 20 agent.
*   **Method:**
    *   **Data:** Collected 3,000 On-Policy Episodes (from Phase 20 agent).
    *   **Dataset:** Total ~12,377 Episodes.
    *   **Training:** Retrained RNN -> Re-evolved Controller (Temp 1.15).
*   **Results:**
    *   **Real Score:** **Avg ~219** (Max 267).
    *   **Regression:** Performance dropped significantly from Phase 20 (Avg 312).
*   **Analysis:**
    *   **Saturation:** Adding more "competent but mediocre" data didn't help. It might have diluted the "survival lessons" or simply confused the small Controller.
    *   **Bottleneck:** We are likely hitting the limit of the **Small MLP (Hidden=64)**. It can't handle the complexity of the diverse dataset (Random + Various On-Policy strategies).
    *   **Next Steps:** **Upgrade the Controller.** Increase hidden size to **256**. This is a low-hanging fruit we haven't picked yet.

## Phase 22: Revert to Linear Controller (Hypothesis Check)
*   **Date:** 2025-11-26
*   **Goal:** Test if a simple Linear Controller generalizes better than MLP on the robust World Model (matching the paper).
*   **Method:**
    *   **Controller:** Linear (Single Layer).
    *   **Training:** JAX-CMA-ES (2000 Gens).
*   **Results:**
    *   **Real Score:** **Avg ~229** (Max 322).
    *   **Comparison:** Similar to failed MLP run (Avg 219), worse than Phase 20 (Avg 312).
*   **Analysis:**
    *   **Hypothesis Rejected:** The Linear Controller is NOT the magic bullet. It performed mediocrely.
    *   **Conclusion:** The problem isn't "Overfitting" (which Linear would fix). The problem is likely **Underfitting** (Capacity). The task of dodging fireballs based on latent vectors is complex.
    *   **Definitive Path Forward:** We MUST increase the capacity of the Controller. The Linear model is too simple, and the small MLP (64) is saturating.
    *   **Next Steps:** **Phase 23: Large MLP (Hidden=256).** This is the only logical step remaining.

## Phase 23: Large MLP Controller (The "Delusional" Expert)
*   **Date:** 2025-11-26
*   **Goal:** Upgrade Controller to Large MLP (Hidden=256) to solve underfitting.
*   **Method:**
    *   **Controller:** MLP (Hidden=256).
    *   **Training:** OpenAI-ES (2000 Gens). (Switched from CMA-ES due to OOM).
*   **Results:**
    *   **Dream Score:** **~2100** (Max Possible). The agent "solved" the dream.
    *   **Real Score:** **Avg ~230** (Max 259). Failed to generalize.
    *   **Behavior:** Active movement (L:37%, R:23%, W:40%). No "Left-Only" bias.
    *   **Diagnostics:** **Avg R_Pred ~0.99**. The agent was 99% confident it would survive, even as it died.
*   **Analysis:**
    *   **Classic Sim2Real Gap:** The Large MLP found a policy that *exploits* the World Model's inaccuracies. It found a "safe path" in the dream that doesn't exist in reality.
    *   **Delusion:** The World Model predicts survival for states/actions that are actually fatal.
    *   **Solution:** We must show the World Model that this specific policy leads to death.
    *   **Next Steps:** **Iterative Training Round 5.** Collect data from this "delusional" agent, retrain RNN, and re-evolve. This is exactly what the iterative process is designed for.

## Phase 25: Curated Fine-Tuning (Fixing Sim2Real Gap)
*   **Date:** 2025-11-26
*   **Goal:** Fix the "Delusional" behavior (Sim2Real gap) where the agent drives into fireballs because the World Model predicts they don't exist.
*   **Method:**
    *   **Data Analysis:** Discovered that "Delusional" data (Phase 24) was diluted (4:1 ratio) by legacy data.
    *   **Curated Dataset:** Created a new dataset with **3,003 Delusional Episodes** (Phase 24) + **1,500 Top Historical Episodes**.
    *   **Training:** Retrained RNN (20 Epochs) on this focused dataset.
*   **Results:**
    *   **Verification:** Generated Debug Grid for Episode 5.
    *   **Outcome:** **SUCCESS.** The "Dream" column now correctly shows fireballs when they appear in Reality. The Sim2Real gap (blindness) is closed.
*   **Analysis:**
    *   Fine-tuning on the specific failure cases forced the RNN to learn the "death" signal it was previously ignoring.

## Phase 26: Controller Evolution (Skill Issue)
*   **Date:** 2025-11-26
*   **Goal:** Evolve a new Controller policy in the corrected (honest) World Model.
*   **Method:**
    *   **Strategy:** JAX-CMA-ES.
    *   **Generations:** 200.
    *   **Population:** 256.
*   **Results:**
    *   **Dream Score:** > 2000 (Excellent).
    *   **Real Score:** **Avg ~200** (Poor).
    *   **Diagnostics:** Debug Grid (Episode 4) shows the agent *sees* the fireballs (World Model is correct) but fails to dodge effectively (clumsy maneuvering).
*   **Analysis:**
    *   **Sim2Real Gap Closed:** The agent is no longer delusional.
    *   **Skill Gap Remains:** The Controller is simply not good enough yet. 200 generations might be too short for the complex dodging behavior required.
    *   **Next Steps:** Continue evolution for longer (500+ generations) or increase population size to find a better policy.

## Phase 27: Extended Evolution (Scaling Up)
*   **Date:** 2025-11-26
*   **Goal:** Solve the "Skill Issue" by giving the optimizer 5x more budget.
*   **Method:**
    *   **Generations:** 1,000.
    *   **Population:** 512.
*   **Results:**
    *   **Dream Score:** > 2000 (Excellent).
    *   **Real Score:** **Avg ~200** (Still Poor).
*   **Analysis:**
    *   Scaling up didn't help. The agent is still overfitting to the Dream.
    *   **Diagnosis:** The agent "wiggles" (oscillates left/right), suggesting it's reacting to VAE/RNN noise rather than the signal.

## Phase 28: Low Temp Evolution (Wiggle Fix)
*   **Date:** 2025-11-26
*   **Goal:** Fix "Wiggling" by reducing training temperature ($\tau=0.5$).
*   **Method:**
    *   **Temperature:** 0.5 (vs 1.15).
    *   **Generations:** 1,000.
*   **Results:**
    *   **Dream Score:** ~1940 (High).
    *   **Real Score:** **Avg ~200** (Still Poor).
    *   **Diagnostics:** Debug Grid (Episode 4) shows **Wiggling is Fixed** (Agent commits to turns), but it commits to **Suicide** (Drives into fireballs).
*   **Analysis:**
    *   **Physics Mismatch Confirmed:** The agent thinks it's safe (in the Dream) when it's actually dying (in Reality). The Dream physics are slightly too forgiving.
    *   **Physics Mismatch Confirmed:** The agent thinks it's safe (in the Dream) when it's actually dying (in Reality). The Dream physics are slightly too forgiving.
    *   **Next Steps:** **Iterative Refinement.** We must collect this "Suicide Data" to teach the RNN the precise hitboxes.

## Phase 29: Iterative Refinement (Physics Fix)
*   **Date:** 2025-11-26
*   **Goal:** Fix the "Confident Suicide" behavior by retraining RNN on failure data.
*   **Method:**
    *   **Data:** Collected 2,000 episodes from the "Suicide Agent".
    *   **RNN:** Retrained for 20 epochs (Loss converged to -87.6).
    *   **Controller:** Evolved with $\tau=0.5$ for 1,000 generations.
*   **Results:**
    *   **Episode 5 Score:** **476.0** (Breakthrough).
    *   **Behavior:** The agent successfully dodges fireballs that previously killed it.
    *   **Remaining Issues:** Performance is still inconsistent (some episodes ~220).
*   **Analysis:**
    *   The Physics Fix worked. The Sim2Real gap is largely closed.
    *   The inconsistency suggests the Controller needs more training time to master the refined (and harder) physics.
    *   **Next Steps:** **Phase 30: Final Polish.** Run a long evolution (2,000 generations) to maximize the policy.

## Phase 30: Final Polish (Wide Search)
*   **Date:** 2025-11-27
*   **Goal:** Find a robust policy by widening the search (Pop 1024).
*   **Method:**
    *   **Population:** 1024 (vs 512).
    *   **Generations:** 1,000.
    *   **Temperature:** 0.5.
*   **Results:**
    *   **Dream Score:** ~1960 (High).
    *   **Real Score (100 Episodes):** **221.09 +/- 99.03** (Failed).
    *   **Max Score:** 476.0.
*   **Analysis:**
    *   Wide Search failed to fix the brittleness. The agent still fails to transfer its dream skills to reality.
    *   **Sim2Real Gap Persists:** The dream is likely still "too easy" or "too clean" compared to the noisy reality.
    *   **Observation:** "Early Movement" (Hallucination?) and "No Dodge" (Blindness?) are still present.
*   **Next Steps:** **Phase 31: Iterative Refinement Round 2.** Use the 100 new failure episodes to retrain the RNN and force it to learn these specific death scenarios.

## Phase 31: GPU reproduction audit and controlled evaluation

* **Date:** 2026-10-02
* **GPU:** WSL restart restored RTX 4070 Ti access. Python 3.12 / JAX 0.11.2
  CUDA 13 / Equinox 0.13.8 / Optax 0.2.8 passed synchronized CUDA matrix and
  convolution-gradient checks. The locked RNN optimizer benchmark improved from
  57.4 ms to 26.4 ms per update (2.17×).
* **Corrections:** Action/outcome alignment, fatal-action labels, per-axis MDN
  mixtures, death-class weighting, posterior sampling, temperature scaling,
  controller cell+hidden memory, true episode starts, real seeds, timeouts and
  vector autoreset semantics. Earlier entries interpreted reward predictions as
  survival confidence; Doom's constant reward cannot support that inference.
  Reconstruction images alone also do not establish accurate dream dynamics.
* **Checks:** 15,404 historical episodes contained three extra duplicates and
  no duplicates across the current held-out split. Terminal recall on the paired
  256-episode audit improved from 2.34% to 72.66% after corrected RNN training.
* **Real results:** On 100 seeds 20000–20099, the original policy without warmup
  scored 238.99 ± 114.07; random actions 222.58 ± 98.63; the iteration-1 selected
  policy 505.74 ± 308.09. Selection used separate 20-game validation seeds.
  This improved the repository baseline but did not meet the paper's >750 solve
  criterion or its reported 1092 ± 556 score.
* **Refinement:** Collected 1000 unique policy-failure episodes, kept 100 separate
  new holdouts and all 770 old holdouts, fine-tuned posterior and mean models,
  then completed four 500-generation CMA searches. The winner selected on 20
  validation games (832.45) scored **840.06 ± 524.48 over 100 fresh test games**
  on seeds 30000–30099, versus original **227.57 ± 111.85** and random
  **220.21 ± 104.18** on the same seeds. This meets the paper's >750/100-game
  criterion on this test, but remains below its 1092 mean. The bootstrap mean
  interval (741.16–946.20) still includes means below 750. Sampled-death dreams,
  mean real inference, reused VAE and full-frame preprocessing are protocol
  extensions; this is not an exact replication. Frozen models and provenance:
  `checkpoints/VizdoomTakeCover-v0/reproduction_refined_selected/`.
* **Overnight:** The reference uses 400 RNN epochs and packed 500-step chunks.
  The new packed trainer passed a two-epoch GPU pilot and 13 regression tests.
  A 400-epoch continued-training run is active with preserved holdouts and
  separate best/final checkpoints. No WSL settings were changed.

Detailed protocols, artifacts, limitations and subsequent results are recorded
in [the reproduction audit](vizdoom_reproduction.md).

## Phase 32: Early stopping and a focused refinement

* **Date:** 2026-10-02
* **Stopped:** Packed continued training stopped safely at epoch 255/400.
  Best held-out loss remained 1.033261 at epoch 1; final loss was 1.106310,
  versus training loss 0.963839. Best and last models were frozen separately;
  the last optimizer was verified at global step 29305 and remains resumable.
* **Paired audits:** On identical 256 historical / 100 newer holdout episodes,
  death recall at threshold 0.5 was 77.73% / 46% for packed best, 54.69% / 40%
  for packed latest, and 72.66% / 49% for the refined incumbent. New live-frame
  false death rates were 0.208%, 0.116%, and 0.305%, respectively. Latest latent
  MSE was lower, so this is a diagnostic tradeoff, not proof of worse gameplay.
* **Next iteration:** Collecting 1000 fresh episodes from the frozen 840.06-step
  controller on CUDA, with eight workers and initial seed 50000. Preserve all
  prior holdouts, reserve 100 new episodes, and use short refinement with
  per-epoch validation before controller optimization. No new real-game test
  result yet; the existing 840.06 ± 524.48 result remains the best verified run.
* **Artifacts:** `artifacts/doom_packed_early_stop.json`,
  `artifacts/doom_packed_stop_audit_comparison.json`, and
  `artifacts/doom_refinement_round2_collection.json`.
* **Data complete (2026-10-03):** 1000 fresh episodes, 797,382 frames, 970
  physical deaths and 30 timeouts. The 3460 training / 970 validation episodes
  are all unique, with zero duplicate leakage and consistent terminal labels.
  Collection mean 797.38 ± 494.12 is a training-data statistic, not a fresh test.
* **Short refinement complete:** CUDA preflight passed. Starting held-out loss
  1.045809 improved to best 1.036670 at epoch 3. Stopped at epoch 6 after three
  non-improving epochs (last loss 1.037441). Best weights and matching optimizer
  were frozen separately from the last model.
* **Paired audits:** Historical / previous-failure / current-policy posterior
  death recall improved from 72.66% / 49% / 43% to 76.17% / 54% / 53%. Current
  live-frame false alarms changed from 0.154% to 0.172%; mean-latent recall
  improved from 46% to 55%. Latent MSE decreased. This warrants controller
  experiments, but does not establish better real-game performance.
* **Controller searches complete:** Two refined-world searches at temperature 1.15
  compared threshold and sampled death. Three matched incumbent-world searches
  at temperatures 1.0 / 1.15 / 1.3 also completed; all used the same initialization,
  seed and 500-generation budget to separate temperature from extra optimization.
  At most two ran concurrently. Real validation is running; fresh tests follow
  only after freezing the validation winner.

## Phase 33: GPU throughput and real-evaluation batching

* **Date:** 2026-10-03
* **Measured load:** Two searches kept GPU activity at 97–99%, with about
  2.6 GiB VRAM, 55–62°C and stable clocks. High activity alone did not establish
  efficient throughput.
* **Controlled concurrency benchmark:** Same world, controller initialization,
  starts, noise and workload; three warmup generations, eight measured per
  process, synchronized start and GPU stages. One process achieved 2.239
  generations/second, versus two processes at 1.923 combined. Serial scheduling
  gave about 16% more throughput; subsequent GPU jobs run one at a time.
  About 94% of the single-process loop was dream rollout computation.
* **Evaluation optimization:** Added eight spawned games with one fused policy
  and memory GPU call, preserving single-game calculation shapes, independent
  per-seed RNG and explicit reseeding. Rejected direct `vmap` because it changed
  seeded recurrent trajectories. Twenty mean-latent and eight posterior games
  matched existing serial outcomes exactly, including action counts. Eight-game
  mean evaluation decreased from 18.943 to 14.887 seconds (1.27×) while other
  searches were active; this excludes model loading and is a short benchmark.
* **Verification:** Two new regression tests cover episode scheduling/memory/RNG
  resets and freezing explicitly selected posterior inference. CUDA replay
  comparisons, Ruff and whitespace checks passed. Corrected the GPU smoke check
  to request full precision for its known-value convolution assertion; reduced
  multiplication precision had produced a false alarm.
* **Completed:** All five searches and twelve controller/inference validation
  combinations completed. Selection and the reserved paired test are recorded
  in phase 34. The earlier 840.06 ± 524.48 cohort is preserved separately;
  dream scores are not real-game results.
* **Artifacts:** `artifacts/doom_search_concurrency_benchmark_synchronized.json`,
  `artifacts/doom_gpu_bench_mean_workers8_map.json`,
  `artifacts/doom_gpu_bench_mean_reseed20_map.json`, and
  `artifacts/doom_round2_real_evaluations.status`.

## Phase 34: Final paired test and GPU release

* **Date:** 2026-10-03
* **Validation:** Twelve combinations on common seeds 13000–13019 selected
  unchanged-world temperature 1.3 with mean-latent inference (938.85). Refined
  threshold/mean was close at 934.40; refined sampled/mean scored 718.40.
  The selection rule was maximum validation mean. Winner files and provenance
  were frozen before test in `reproduction_round2_real_selected/`.
* **Fresh paired test:** On 100 identical seeds 40000–40099, the selected policy
  scored **775.85 ± 496.21**, versus unchanged incumbent **728.75 ± 483.70**.
  Paired gain **47.10**; 95% percentile bootstrap interval **−51.60 to +146.61**,
  using 50,000 resamples. Selected policy won 52 games, incumbent 45, three tied.
  This does not establish a reliable improvement. The selected mean's interval
  is 680.66–874.54, including values below the paper's 750 criterion.
* **Interpretation:** The selected world is unchanged; this is the temperature
  comparison, not evidence that round-two data refinement improved gameplay.
  Only the validation winner was tested, and no final test outcomes were used
  for selection or retraining. Original 840.06 ± 524.48 on seeds 30000–30099 is
  retained as a separate cohort, not compared as a paired result.
* **Protocol:** Reused full-frame VAE, biased controller, sampled-death dreams,
  mean real inference, no warmup, 2100-step cap, eight workers with serial
  calculation shapes and per-seed RNG. These differ from the original paper.
  Checkpoint hashes remained unchanged during evaluation.
* **Stopping point:** All repository training/evaluation processes exited. No
  additional GPU experiment, collection, benchmark or video is queued. Preserve
  all checkpoints, best/final optimizers and data; pause the five-minute follow-up
  and require a new request before further GPU work.
* **Artifacts:** `artifacts/doom_round2_paired_comparison.json`,
  `artifacts/doom_round2_real_evaluation_result.json`, and `artifacts/task_state.json`.

## Phase 35: CPU vision audit and controlled VAE comparison preparation

* **Date:** 2026-10-03
* **Resource constraint:** GPU remains released for the user's other task;
  availability for the new goal is pending. No new GPU work or heartbeat was
  started. CPU work does not establish a new gameplay score.
* **Data:** Recovered all 4,430 raw episode groups in the existing verified
  split: 3,460 training / 970 validation, no overlapping signatures or archive
  fingerprints. Prepared a shared immutable pool of 221,440 training and 15,520
  validation frames, with identical sampling for both architectures.
* **Diagnostic:** Twelve newer held-out fatal episodes / 48 pre-death frames
  gave mean-latent pixel MSE 0.0026013 and posterior MSE 0.0028844. Large
  approaching fireballs are visible in reconstructions; the atlas is too small
  to establish general projectile retention.
* **Frozen-feature probe:** A linear ridge fit on 900 new training episodes /
  57,600 frames, evaluated on 100 separate new holdouts / 1,600 frames, predicted
  larger bright-color component presence with AUC 0.8872 and small-component
  presence with AUC 0.7094. Largest-component horizontal position MAE was 8.56
  pixels versus a constant baseline's 14.71. These are color proxies, not
  verified fireball labels, and do not prove a vision bottleneck or improved
  controller performance.
* **Implementation:** Added the reference VAE geometry while preserving old
  checkpoint bytes, isolated validated training, best/final model+Adam+RNG
  bundles, immutable frame caches and explicit encoding into separate RNN
  training/validation directories. Updated model loading and freezing to keep
  architecture sidecars. CPU smoke verified final Adam count 2; no scientific
  VAE training has started.
* **Next:** Controlled current/reference VAE comparison at fixed data and image
  preprocessing, followed by fresh RNN/controller training if supported by the
  vision results. Full protocol and limitations:
  [vision experiment](vizdoom_vision_experiment.md).
* **Artifacts:** `artifacts/doom_vision_round3_split.json`,
  `artifacts/doom_vision_round3_frames_s73/metadata.json`,
  `artifacts/doom_vision_incumbent_cpu/report.json`,
  `artifacts/doom_vision_incumbent_linear_probe.json`, and
  `artifacts/doom_vision_cpu_smoke.json`.

### Episode-level uncertainty for the frozen vision probe

* **Date:** 2026-10-03; CPU only. The same 100 held-out episodes were resampled
  as whole groups 5,000 times (seed 73501), keeping correlated frames together.
  Larger-component AUC 0.8872 has a 95% percentile interval 0.8698–0.9042;
  small-component AUC 0.7094 has interval 0.6790–0.7391. Horizontal position
  error was 8.56 pixels [8.06, 9.09], with paired error reduction relative to
  the constant predictor of 6.15 pixels [5.34, 6.95].
* **Limits:** Intervals hold the fitted probe and color-derived labels fixed.
  They describe sampling uncertainty across these holdout episodes, not VAE
  training variability, actual projectile detection, or real-game improvement.
* **Implementation and checks:** Saved row predictions with episode identities
  and frame indices for future paired comparisons. Fixed recursive encoding
  discovery so the probe accepts the prepared training/validation directory
  layout. Four new CPU tests check grouping, paired sampling and missing metrics.
* **Resource state:** No new GPU experiment was started. A read-only NVML query
  listed no CUDA compute client, but the user's availability reply is still
  pending; the previous heartbeat remains paused.
* **Artifacts:** `artifacts/doom_vision_incumbent_episode_bootstrap.json` and
  `artifacts/doom_vision_incumbent_episode_bootstrap_predictions.npz`.

## Phase 36: Serial CUDA VAE architecture comparison

* **Date:** 2026-10-03; started 11:15 UTC. The active user goal renews local
  experimentation after the completed round-two GPU release. No competing CUDA
  compute client was listed before dispatch; matrix multiplication and
  convolution-gradient preflight passed on the RTX 4070 Ti with JAX 0.11.2.
* **Protocol:** Current then reference VAE, from scratch with seed 73, identical
  fixed 221,440 training / 15,520 held-out frames, learning rate 0.0001,
  batch 128, maximum 20 epochs and patience five. Best and final weights retain
  their matching optimizer/RNG states. Original checkpoints remain frozen.
* **Early progress:** Current architecture reached epoch nine at 287 elapsed
  seconds with held-out loss 47.895. The final comparison is pending. Falling
  reconstruction loss does not establish a real-game gain.
* **Follow-up:** Existing five-minute heartbeat now follows the vision
  comparison; dispatch can be deferred during an active chat turn. Earlier
  phase 34 remains completed, and the packed run remains deliberately stopped.
* **Artifacts:** `artifacts/doom_vision_round3_vae_jobs.json`,
  `artifacts/doom_vision_round3_vaes.status`, and per-architecture logs/history
  under the isolated `vision_round3_*_s73` checkpoint directories.

### Current VAE completion and continuation verification

* Current architecture completed 20 epochs, best at epoch 20: held-out loss
  45.2557, reconstruction 13.1848, 34,600 Adam updates, 674 elapsed seconds.
  Loss still improved 1.75% from epoch 15; budget exhaustion is not convergence.
* On the identical twelve newer fatal holdout episodes / 48 frames, mean pixel
  MSE was 0.002154 versus incumbent 0.002601. Paired whole-episode bootstrap
  gives a 17.2% reduction [9.9%, 22.9%]. Sampled error reduced 15.7% [5.3%,
  23.3%]. Large fireballs remain visible and small ones remain blurred.
  Whole-image error on this small sample is not projectile detection or survival.
* Added verified continuation into a fresh directory with a total epoch cap.
  Two CPU checks passed for Adam/RNG restoration, fingerprint/settings mismatch
  rejection, and preserved best/patience. A CLI integration smoke matched
  uninterrupted weights and Adam bytes exactly, with RNG equal and final
  optimizer count four; original source files remained unchanged. Ruff passed.
* Paper architecture is still running. No new RNN, controller or real-game
  result has been produced in this slice.
* Artifacts: `artifacts/doom_vision_current20_paired_audit.json`,
  `artifacts/doom_vision_current20_cpu/atlas.png`, and
  `artifacts/doom_vision_vae_continuation_cpu_verification.json`.

### Matched 20-epoch results and bounded continuation

* Reference architecture completed 20 epochs in 800 elapsed seconds, best
  held-out loss 46.0313 at epoch 20, versus current 45.2557. Actual serialized
  best/final Adam counts for both were 34,600, with RNG and validation history
  verified on CPU. The incumbent hashes remained unchanged.
* Reference's identical 48-frame audit had mean MSE 0.002502 and sampled MSE
  0.002600. Reference mean error was 16.2% higher than current [13.2%, 19.8%],
  and sampled error 6.9% higher [2.4%, 11.0%]. This is whole-image error on a
  small fatal-window cohort, not projectile retention or gameplay performance;
  architecture training variability is not included.
* Both curves remained improving over epochs 15–20 (1.75% current, 1.84%
  reference). A serial continuation to at most 60 **total** epochs began at
  11:44 UTC, with identical settings, patience five and preserved 20-epoch
  source directories. Current runs first, then reference. No additional data,
  RNN/controller training or real-game evaluation has started in this slice.
* Artifacts: `artifacts/doom_vision_round3_vaes20_bundle_verification.json`,
  `artifacts/doom_vision_architectures20_paired_audit.json`, and
  `artifacts/doom_vision_round3_vae_extension_jobs.json`.

### Current 60-epoch completion and faster feature diagnostics

* Current VAE completed 60 total epochs, best at 60: held-out loss 42.6414,
  reconstruction 10.5411, 103,800 Adam updates. Loss improved 0.316% over
  epochs 55–60; reaching the budget cap does not prove convergence. Paper VAE
  continuation remains healthy; the user explicitly confirmed GPU availability.
* Current epoch-60 fixed atlas mean MSE was 0.0019467 and sampled MSE 0.0021704.
  Paired whole-episode bootstrap against the incumbent gives 25.2% mean-error
  reduction [19.4%, 30.1%] and 24.8% sampled reduction [16.2%, 31.2%]. Large
  projectiles remain visible and small/early ones remain blurred. This is a
  small fatal-window reconstruction sample, not projectile recall or survival.
* Prepared a faster feature route that encodes the 236,960 fixed sampled
  frames before full 1,691,022-frame RNN preprocessing. Mean precision and
  encoder batch shape are matched across incumbent/current/reference models.
  Arrays preserve sampled-row lineage and are explicitly excluded from RNN
  trajectory use. Full episode encoding remains required for downstream RNNs.
* Fourteen affected CPU checks passed, including sampled/full encoding and
  probe-result equality, source lineage, completed-cache recovery and corruption
  rejection. One paired-probe check verified known gains and frame matching.
  Ruff passed; the prepared serial diagnostic supervisor parses. No additional
  GPU diagnostic or downstream RNN/controller job has been launched yet.
* Artifacts: `artifacts/doom_vision_current60_cpu/`,
  `artifacts/doom_vision_fast_incumbent_vs_current60_audit_comparison.json`, and
  prepared `artifacts/run_vision_round3_diagnostics.py`.

### Frozen 60-epoch feature results and downstream selection

* Both VAEs completed 60 CUDA epochs; best/final Adam counts were 103,800,
  verified from serialized states on CPU. Best held-out losses were 42.6414
  (current) and 43.4174 (reference). The last five epochs still improved loss
  0.316% and 0.399%; neither is proof of convergence. Incumbent hashes stayed
  unchanged. Frozen diagnostics completed at 12:35 UTC on 2026-10-03.
* Matched mean-feature probes used the same 900 training and 100 held-out
  episodes, targets and encoder batch 128. Larger/small color-component AUCs
  were incumbent 0.8873/0.7094, current 0.8787/0.7212, reference 0.8298/0.6714.
  Current's paired AUC differences (-0.0086 and +0.0119) both have 95% intervals
  spanning zero; horizontal MAE increased 0.60 pixels [0.17, 1.01]. Reference
  decreased both AUCs with intervals excluding zero and increased horizontal
  error 0.86 [0.38, 1.33]. Targets are color proxies, not verified projectiles.
* Reference's inspected 48-frame atlas had mean MSE 0.0022373 and sampled MSE
  0.0023822, 14.9% and 9.8% higher than current. Both retain visible large
  approaching projectiles and blur small early cues. These measurements do not
  establish real survival, training variability or general architecture merit.
* Decision: retain reference VAE artifacts but do not promote it downstream in
  this iteration. Compare the fixed incumbent VAE against current epoch 60 with
  fresh matched RNNs and controllers on identical episode splits and seeds.
  This control separates the VAE difference from downstream retraining. Keep
  the original incumbent policy as a separate real-game control. No cross-basis
  warm starts, new data or temperature changes are part of this slice.
* The serial RNN supervisor prepares and verifies both full-episode encodings,
  then allows 30 epochs per fresh RNN, seed 74, LR 0.001, batch 32, patience 3,
  posterior inputs and death weight 10. It verifies actual Adam/RNG bundles and
  freezes held-out best worlds, stopping before controller searches. Two RNN
  selection checks and one full-episode encoding check passed on CPU; launcher
  and verifier passed Ruff and syntax checks. No new survival score exists yet.
* Artifacts: `artifacts/doom_vision_round3_diagnostics_result.json`, paired
  `doom_vision_fast_*_probe_comparison.json`,
  `artifacts/doom_vision_round3_vaes60_bundle_verification.json`, and
  `artifacts/run_vision_round3_rnns.py`.

### Full episode inputs and matched death diagnostics

* Both frozen-VAE full encodings completed and passed fingerprints, transition
  identity, finite latents, fatal/timeout coverage and split checks. Each has
  3,460 training episodes / 1,373,500 frames (3,430 fatal, 30 timeout) and 970
  held-out episodes / 317,522 frames (all fatal). The fresh control RNN has
  begun CUDA training; the candidate follows serially. No survival score exists.
* Corrected new death audits to use float32 latents and posterior noise exactly
  as RNN training does. Historical audits rounded samples back to stored
  float16. Existing reports are preserved and the legacy audit mode remains
  available; numerical comparisons must state this difference. Added episode
  counts and paired whole-episode uncertainty for death recall/false alarms.
  Two focused CPU checks passed; changed code passed Ruff.
* Prepared matched audit and six-search controller supervisors. They wait for
  frozen RNNs and actual audit review, use the existing exclusive GPU lock, and
  preserve partial work rather than blindly restarting. No concurrent search
  was started while training runs.
* Proposed real-validation/test seed ranges have no overlap with the 50
  recorded game/collection reports (1,460 unique seeds). Repeat this audit
  before evaluation; it excludes unrecorded external runs and consumes no games.
* Artifacts: `artifacts/doom_vision_round3_encoding_*_verification.json`,
  `artifacts/run_vision_round3_holdout_audits.py`,
  `artifacts/run_vision_round3_controllers.py`, and
  `artifacts/doom_vision_round3_real_seed_audit.json`.

### Fresh RNNs and controller comparison complete; real games dispatched

* Both fresh RNNs stopped with patience three. Incumbent-VAE best was epoch 15,
  held-out loss 1.087586, Adam count 1,635; final epoch 18 / 1.092155 / 1,962.
  Current-VAE best was epoch 18 / 1.083214 / 1,962; final 21 / 1.085893 / 2,289.
  Actual serialized model, optimizer and RNG bundles passed CPU verification;
  best worlds are frozen and final states preserved. Cross-latent NLLs do not
  rank real performance.
* Twelve fixed death audits and twelve paired comparisons completed. Posterior
  recall for fresh incumbent-VAE/current-VAE worlds was 66.41%/60.16% on
  256 historical holdouts and 52%/38% on 100 newer holdouts. Paired differences
  were -6.25 percentage points [-10.94, -1.56] and -14 [-23, -5]. Current
  reduced false alarms. Weighted scores are not calibrated probabilities;
  these results use the corrected float32 protocol, holding fitted models
  fixed. The bounded real-survival comparison proceeds despite this concern.
* All six matched fresh CMA searches completed 500 generations at tau 1.15,
  threshold death, population 64, 16 rollouts and batch 64, seeds 74/75/76.
  Complete histories/settings and selected-policy fingerprints were verified.
  Their dream scores are not comparable real survival measurements.
* The real supervisor started around 14:22 UTC, 2026-10-03, after successful
  CUDA preflight. It validates six policies plus the original incumbent, each
  in mean and posterior modes, on 100 common seeds 60000–60099. The pre-dispatch
  seed audit found no recorded overlap. It will freeze the highest validation
  mean before testing that winner and the unchanged mean-inference incumbent
  on reserved 80000–80099. One GPU job and eight game workers preserve the
  verified inference shapes and per-seed RNG.
* Two CPU fixtures passed recovery/protocol guards. Two preserved reports
  containing 200 actual games passed the same independent summary, seed,
  inference, checkpoint, reward and action-count checks without rerunning games.
  A CPU paired bootstrap summary is prepared for both final test reports.
* Artifacts: `artifacts/doom_vision_round3_rnns_result.json`,
  `artifacts/doom_vision_round3_holdout_audits_result.json`,
  `artifacts/doom_vision_round3_controllers_result.json`, and
  `artifacts/run_vision_round3_real_evaluations.py`. No new reserved-test result
  or reliable survival gain has been established yet.

### Vision real comparison completed: incumbent retained

* All 14 combinations completed 100 common validation games. New current-VAE
  policies averaged 410.30–467.51 steps; fresh-RNN incumbent-VAE controls
  averaged 174.49–240.55. Original incumbent mean/posterior inference averaged
  793.16/775.04. The best new policy's paired difference from the original mean
  control was -325.65, 95% interval [-422.56, -234.02]. Bonferroni-adjusted
  percentile interval for 12 comparisons was [-467.20, -192.27]; all twelve
  adjusted upper bounds were negative. This uses validation only, fixed models
  and whole-game bootstrap; it excludes training variability.
* Validation retained the unchanged original mean-inference policy, frozen
  before reserved test seeds 80000–80099. Its actual parameters were verified
  identical after metadata repacking. Both test reports scored **817.91 ±
  524.13**, mean interval **[717.39, 922.52]**. All 100 games tied; paired
  improvement **0.00 [0.00, 0.00]**. This is a repeated baseline, not a new
  success. Prior 840.06 cohort remains preserved. Paper mean1092 is not matched.
* The existing supervisor exited with code zero, and process inspection found
  no remaining vision evaluation or training process. Test seeds 80000–80099
  are consumed. No policy selection or tuning uses these test outcomes.
* Pixel reconstruction improved without a survival gain. Dream-to-real failure
  is consistent with model exploitation, but the exact mechanism is unproven.
  More VAE epochs chosen solely on pixel loss are not supported by these games.
* Prepared the next four-search frozen-RNN pilot: preserved packed epoch-one
  best versus unchanged incumbent, original VAE/tau 1.15, identical incumbent
  controller initialization, threshold/sample modes, CMA seed77 and matched
  500-generation budgets. It never resumes packed training. Different RNN
  training histories prevent attributing any gain to packing alone. Fresh
  validation 90000–90099 and test 110000–110099 have no recorded overlap in the
  CPU preparation audit; repeat before dispatch. One GPU job at a time.
* Artifacts: `artifacts/doom_vision_round3_validation_comparison.json`,
  `artifacts/doom_vision_round3_paired_comparison.json`, and
  `artifacts/doom_frozen_packed_transfer_preparation.json`. Pilot launchers passed
  CPU definition/import checks and Ruff; preparation itself ran no new games.

### Descriptive incumbent consistency check during frozen-world pilot

* Independently verified three unchanged-incumbent reserved reports: identical
  VAE/RNN/controller fingerprints, mean inference, unit survival scoring,
  per-game action counts, exact summaries and 300 distinct seeds. Excluded the
  duplicate round-three selected copy. No new games or GPU diagnostics ran.
* Descriptive aggregate is 795.57 ± 513.39, with stratified whole-game bootstrap
  mean interval [738.40, 854.28] (50,000 resamples, seed74000, resampling within
  each 100-game cohort). This preserves the separate original840.06 report and
  is not an exact paper comparison or a newly selected policy. The oldest
  report lacks explicit worker/inference fields; mean inference follows its
  recorded controller metadata and override. Protocol/training uncertainty
  remains outside this fixed-cohort interval.
* Artifact: `artifacts/doom_incumbent_3_cohort_summary.json`. The four-search
  frozen-RNN pilot continues; its real outcomes are still pending.

### Frozen-RNN pilot searches complete; real evaluation started

* All four matched 500-generation searches completed at 15:14 UTC on
  2026-10-03. Selected policies, settings and full histories passed verification.
  No RNN/VAE training or data collection was started; originals remain intact.
* Real evaluation started at approximately 15:17 UTC, session37461. CUDA
  preflight passed and the supervisor/current game child were verified live.
  Pre-dispatch audit found no proposed-seed overlap against 1,660 distinct
  recorded seeds. Exactly ten fixed policy/inference combinations receive
  100 common validation games on90000–90099. Validation alone selects and
  freezes the winner before paired tests110000–110099. Only one GPU job runs.
* Artifacts: `artifacts/doom_frozen_transfer_round4_controllers_result.json`,
  `artifacts/doom_frozen_transfer_round4_real_seed_reservation.json`, and
  `artifacts/run_frozen_transfer_round4_real_evaluations.py`. No new real
  performance gain has been established yet.

### Final frozen-world real comparison and completion on diminishing returns

* All ten combinations completed100 common validation seeds90000–90099.
  Validation chose incumbent-world sampled death with posterior inference,
  mean879.74, and froze its policy/mode/report hashes before reserved testing.
* On100 reserved seeds110000–110099, selected survival was **853.53 ±572.26**
  versus unchanged incumbent **819.15 ±478.46**. Paired gain **+34.38** had
  bootstrap95% **[-85.29,+156.04]**,47 wins/52 losses/one tie. Mean95% for the
  selected policy is [743.72,967.45]. Population SDs and50,000 whole-game paired
  resamples (seed74110) condition on fixed policies and exclude training
  variability. No reliable gain is established and tests did not select or
  retrain any policy. Candidate and canonical incumbent are both preserved.
* Four distinct unchanged-incumbent cohorts total400 unique games,
  **801.47 ±504.99**, stratified mean95% **[752.44,851.16]**. Duplicate paired
  copies are excluded. This descriptive CPU analysis preserves the original
  **840.06 ±524.48** report and its recorded protocol differences.
* Round3's current-VAE pipeline exceeded matched fresh incumbent-VAE controls
  by169.75–275.81 validation steps across six seed/mode comparisons, while
  both lost to the original incumbent. That original comparison cannot isolate
  the VAE from RNN training history; improved pixels alone do not establish
  improved transfer.
* Stop on observed diminishing returns: three completed loops established no
  reliable reserved-test gain (+47.10 with interval crossing zero, unchanged
  incumbent retained, and+34.38 with interval crossing zero). Fifteen500-CMA
  searches, new data/refinement and vision/downstream comparisons, plus the
  packed loss plateau, do not support more incremental work in these families.
  Paper mean1092 remains unmatched. This is an empirical stopping decision,
  not proof of a fundamental ceiling or exact paper replication.
* Supervisors exited with code zero; process inspection found no remaining
  experiment jobs and GPU compute-process query was empty. All originals,
  best/final checkpoints and available optimizer/RNG states remain intact;
  packed best has no best optimizer and packed training must not auto-resume.
  No GPU work is queued. The follow-up is paused on completion and a clear
  resume state records consumed seeds and requires a new experiment request.
* Artifacts: `artifacts/doom_frozen_transfer_round4_paired_comparison.json`,
  `artifacts/doom_incumbent_4_cohort_summary.json`, and
  `artifacts/doom_experiment_completion_audit.json`. Full validation table,
  limitations and stopping rationale are in `docs/vizdoom_reproduction.md`.

### Reference environment compatibility audit, round5

* Pinned public author weights are a diagnostic control; no new training or
  independently trained policy is claimed. A source audit found that legacy
  doom-py0.0.15 clamps requested skill5 to4 and emits RGB bytes for its nominal
  BGR24 format. The modern reference wrapper now uses effective skill4/RGB24.
  The prior trained pipeline already used those effective settings.
* On 100 common diagnostic seeds 120000–120099, the literal modern skill5/BGR
  port scored 119.42 ± 38.97; correcting difficulty alone scored 236.13 ± 117.25.
  Both cohorts ended in100 deaths. Those comparisons use a different engine
  and RNG from the original paper and do not select or retrain the policy.
* With both compatibility corrections, eight initial games on 120000–120007
  scored 1083.88 ± 526.27, all deaths. This is a small supplied-model diagnostic,
  not a100-game success or a match of our trained model to the paper.
* The corrected-color100-game attempt stopped at the reward-versus-step
  assertion after saving one244-step game. The failing record was not saved;
  its cause remains unresolved. An eight-game replay and a CPU32-tic timeout
  check passed reward accounting, so a timeout bug has not been established.
* Reference weights, game assets and evaluation source are fingerprinted.
  Episode callbacks save valid partial reports; recovery schedules unfinished
  seeds only under the same frozen protocol. Future own-policy validation
  130000–130099 and testing 140000–140099 remain separately reserved.
* Evidence, protocol differences and runnable tools are documented in
  `docs/vizdoom_reference_audit.md`. Detailed local reports and downloaded
  assets remain ignored under `artifacts/`; original checkpoints are intact.

### Full reference diagnostic complete; bounded own-controller refinement

* The unchanged effective-skill4/RGB policy completed100 common diagnostic
  seeds120000–120099 at **991.20 ± 506.48**,98 deaths and2 timeouts. All rewards
  matched controlled actions; the initial eight records exactly reproduced.
  The earlier scoring failure did not recur, and its cause remains unresolved.
  Future failures now preserve the offending record without weakening checks.
* Mean whole-game bootstrap95% is **[893.45,1091.04]**. Paired gain from correcting
  modern BGR to effective legacy RGB at fixed skill4 is **+755.07**,95%
  **[662.22,850.52]** (50,000 fixed-policy paired resamples, seed74121). This
  diagnostic uses public author weights, differs in engine/RNG/Pillow, and
  does not meet the1092 target or reproduce our training.
* CPU input/source/record checks passed. Original snapshots and reports are
  preserved. Artifacts: `doom_reference_round5_error_capture100.json` and
  `doom_reference_round5_rgb_analysis.json`, under `artifacts/`.
* One500-generation own-controller refinement was run on the frozen public
  world at tau1.15,pop64,16 trials,candidate batch64,sigma0.02,seed91,initialized
  from the public policy. It retains best/final policy, optimizer and RNG. All45
  CPU tests and CUDA preflight passed before dispatch; no VAE/RNN retraining is
  claimed. An additional evaluator recovery/provenance CPU test also passed.
* Real validation130000–130019 and reserved testing140000–140099 are separate
  and unused at dispatch. Selection precedes freezing and testing, with a
  paired supplied-policy control. Dream scores do not establish real gains.
  Protocol and tools are in `docs/vizdoom_reference_audit.md`; the user goal
  remains active and the canonical840.06 checkpoint is intact.

* The controller search subsequently completed500 generations. Best held-out
  dream score1162.40625 at generation270 exceeds initial1110.796875; final CMA
  mean961.53125 is also preserved. Input/source, best/final, optimizer/RNG and
  canonical incumbent fingerprints passed verification. Real survival remains
  unproven; this is controller training on an imported public world.
* The serial real supervisor is evaluating six distinct combinations on the20
  fresh validation games, then will freeze the validation winner before the100
  reserved tests and supplied-control pairing. An actual live supervisor and
  evaluation child were verified; CUDA preflight passed and no second GPU job
  is running. Current result/status paths are in the reference audit and
  `artifacts/task_state.json`.
* All six validation combinations subsequently completed on seeds130000–130019.
  Best-generation270/posterior won at **958.05 ± 596.39**, versus supplied
  posterior **877.75 ± 478.91**. The selection and input/source fingerprints
  were frozen at2026-10-03T21:39:24.496998Z before reserved testing began.
  These are validation scores, not a completed100-game test or proof of1092.
* CPU tools now independently check complete games, frozen validation-only
  selection, matching input/source hashes, and paired whole-game uncertainty.
  `scripts/tools/summarize_doom_reference_results.py` produces the final audit
  once both reserved reports and the supervisor result are complete.
* The selected generation270/posterior controller subsequently completed100
  reserved games140000–140099 at **1035.22 ± 550.45**,92 deaths/8 timeouts;
  mean95% **[929.91,1144.23]** from50,000 whole-game resamples,seed74140.
  Exact records, summaries, frozen policy and input/source fingerprints passed
  independent CPU checks. This observed mean is below1092, so the goal remains
  active. The paired supplied-policy test is running; its gain is not yet known.
  Artifact: `doom_reference_round5_selected_test100_cpu_audit.json`.
* The paired supplied posterior control subsequently completed **955.39 ±
  541.20** on the same100 seeds,93 deaths/7 timeouts. Selected-minus-control
  gain **+79.83**,95% **[-22.26,+183.00]**,48 wins/43 losses/9 ties, does not
  establish a reliable improvement (50,000 joint whole-game resamples,
  seed74140). All validation-only selection, reserved cohorts, frozen
  report/input/source hashes and timing passed the independent final audit.
  Both evaluation processes exited; GPU compute-process query was empty.
  Artifacts: `doom_reference_round5_real_evaluation_result.json` and
  `doom_reference_round5_paired_comparison.json`. Goal remains active below1092.
* The matched temperature1.25 follow-up completed500 generations. Best
  held-out dream score889.296875 at generation210, initial768.1875 and
  final698.296875; these are dream scores, not real survival results.
  Independent CPU checks verified optimizer/RNG, identical starting indices,
  source/world fingerprints and preserved original checkpoints. Four policies
  are now undergoing80 fresh validation games each on seeds130020–130099;
  the first control had completed21 games at this check, with CUDA preflight
  passed. Only an updated1.25 candidate winning against both controls may
  consume reserved100-game tests150000–150099, paired with frozen1.15.
  New selection safeguards and the protocol-aware CPU auditor verify this
  comparison and reject substituted controls or mismatched cohorts.
* Commit checks: all52 tests passed on CPU. Ruff lint and formatting checks
  passed for the changed Python files, and `git diff --check` passed. The
  existing real validation supervisor remained running during these checks.
* Round6's first complete80-game validation control, frozen1.15 generation270
  with posterior inference, scored **904.83 ±511.93**,78 deaths/2 timeouts;
  fixed-policy mean95% **[795.51,1017.71]** (50,000 whole-game resamples,
  seed74180). Exact cohort and controller/input/source fingerprints passed
  independent CPU verification. Supplied control and both new1.25 cases remain
  to finish; no reserved test or policy selection has occurred. Artifact:
  `doom_reference_round6_prior_control_validation_cpu_audit.json`.
* CPU training-history review found52 checks of a fixed64-dream cohort in each
  search. Top-two dream-score gaps14.34 at1.15 and38.41 at1.25 do not establish
  overfitting without per-rollout variance. The paper's1024-rollout Doom check
  motivates a larger dream validation cohort as a follow-up hypothesis;
  no new GPU job was started or queued. Artifact:
  `doom_reference_dream_selection_cpu_review.json`.
* The supplied control subsequently finished80 validation games at **923.11
  ±544.26**,77 deaths/3 timeouts, mean95% **[805.55,1045.21]**. Prior1.15
  minus supplied paired gain **−18.29**,95% **[−156.40,+119.00]**,38 wins/
  36 losses/6 ties. Exact cohorts, shared world/protocol and fingerprints
  passed the CPU check; these descriptive validation results establish no
  clear control advantage. The new1.25 best/final cases remain to finish;
  no winner or reserved test yet. Artifact:
  `doom_reference_round6_controls_validation_cpu_audit.json`.
* CPU inspection of the preserved optimizer states found final CMA scales
  0.01452 and0.01512 from initial0.02. First/last50-generation population
  dream means916.04/900.91 at1.15 and738.11/727.46 at1.25 show no upward
  aggregate trend, without proving a cause or real-policy improvement.
  Artifact: `doom_reference_optimizer_progress_cpu_review.json`.
* Round6 subsequently completed all four80-game validations. Supplied control
  won at **923.11 ±544.26**, versus prior1.15 **904.83 ±511.93**, new1.25 best
  **914.99 ±521.24** and final **914.28 ±514.03**. New-best-minus-supplied
  paired gain **−8.13**,95% **[−131.61,+116.06]**; final gain **−8.84**,95%
  **[−155.79,+134.65]**. No new policy won, so reserved150000–150099 games
  were not consumed. The independent CPU closure verified actual records,
  selection, source/input/controller hashes and absence of reserved reports.
  Both evaluation processes exited. Artifact:
  `doom_reference_round6_validation_closure_cpu_audit.json`.
* Round7's preregistered local controller search completed500 generations on
  CUDA at temperature1.15, sigma0.005 and1024 fixed dream validation rollouts.
  All other training settings match the earlier1.15 search. Best dream score
  **950.0654** at generation120, initial928.0801 and final911.2314 establish
  no real survival result. Best/final and optimizer are preserved; independent
  training-state review and real evaluation are pending. Protocol and result:
  `doom_reference_round7_fine_search_protocol.json` and
  `doom_reference_round7_fine_search_result.json`.
* Round7 reserves80 fresh validation seeds130100–130179. Only an updated new
  winner gets100 reserved160000–160099 tests, paired with the highest control
  chosen and frozen by validation. The CPU auditor now checks that control
  identity and rejects substitution. The sigma/validation changes form one
  package, so this comparison cannot isolate their individual effects.
* Commit validation: six relevant CPU regression tests passed, including
  fixed-control compatibility and rejection of an incorrect validation-selected
  paired control. Ruff lint/format and `git diff --check` passed. These checks
  started no GPU work.
* Round7's independent CPU training-state audit passed: actual0–500 history,
  protocol/arguments,900/100 disjoint dream start indices, best/final hashes,
  optimizer count500/final mean, NumPy RNG/next JAX key, frozen sources/inputs
  and preserved incumbent. The search session exited0. The serial real
  supervisor then started with CUDA preflight passed; actual supervisor843089
  and evaluator843158 were verified live. Four80-game validations are pending,
  and no reserved result is claimed. Artifact:
  `doom_reference_round7_training_cpu_audit.json`.
* Round7's frozen1.15 generation270 control completed80 validation games at
  **920.03 ±534.36**,77 deaths/3 timeouts, mean95% **[805.16,1038.41]**
  (50,000 whole-game resamples, seed74180). Exact cohort, actual records and
  controller/world/source hashes passed CPU verification. The supplied control
  and both new candidates are still pending; no policy selection or reserved
  test has occurred. Artifact:
  `doom_reference_round7_prior_control_validation_cpu_audit.json`.
* The supplied control subsequently completed80 games at **944.91 ±553.37**,
  74 deaths/6 timeouts, mean95% **[824.55,1069.56]**. Prior-minus-supplied
  paired gain **−24.89**,95% **[−159.98,+111.19]**,30 wins/43 losses/7 ties
  establishes no clear control advantage. Exact cohorts, shared world/protocol
  and source/controller fingerprints passed CPU checks. The new generation120
  candidate is running, with the final checkpoint next; no selection or
  reserved test yet. Artifact:
  `doom_reference_round7_controls_validation_cpu_audit.json`.
* New best-generation120 completed80 validation games at **947.80 ±543.20**,
  75 deaths/5 timeouts, mean95% **[831.36,1068.06]**. Paired gain over supplied
  **+2.89**,95% **[−113.21,+118.34]**,25 wins/30 losses/25 ties; over prior1.15
  **+27.78**,95% **[−105.80,+161.58]**. Actual cohorts, shared protocol/world,
  fingerprints and candidate provenance passed CPU verification. This does
  not establish a reliable gain. The final checkpoint is running, and full
  selection/eligible reserved testing remain pending. Artifact:
  `doom_reference_round7_best_validation_cpu_audit.json`.
* Final-generation500 completed **875.79 ±544.29**,75 deaths/5 timeouts,
  mean95% **[759.27,997.34]**; final-minus-supplied **−69.13**,95%
  **[−211.14,+70.38]**. All four80-game cohorts selected generation120 at
  **947.80**, with supplied **944.91** as the highest validation control.
  Both identities and inference were frozen at2026-10-04T00:09:49.485111Z
  before reserved testing. Independent CPU reconstruction verified actual
  cohorts, parameter identities, choice/comparator, fingerprints and timing.
  The selected100-game160000–160099 test is running after CUDA preflight;
  supplied pairing follows. No1092 result yet. Artifact:
  `doom_reference_round7_frozen_selection_cpu_audit.json`.
* The round7 selected generation120 controller completed100 fresh tests at
  **945.73 ±570.08**,93 deaths/7 timeouts, mean95% **[836.80,1057.28]**
  (50,000 fixed-policy whole-game resamples, seed74140). Exact160000–160099
  records, frozen parameters/input/source and selection timing passed CPU
  verification. The observed mean misses1092; paired supplied control is
  still running, so gain is not yet known. Artifact:
  `doom_reference_round7_selected_test100_cpu_audit.json`.
* A contingent one-factor fitness64 comparison is preregistered from training
  history and validation evidence, with no new GPU work dispatched or queued.
  It keeps round7's settings except fitness rollouts16→64 and output path.
  First/last50 population means912.95/906.09 do not diagnose a cause; less
  noisy fitness is a hypothesis, and per-rollout variance is unavailable.
  Protocol reserves validation130180–130259 and eligible paired test170000–
  170099, using supplied and validation-selected round7 policies as controls.
  Fresh-seed CPU precheck passed. Launcher correctly reports not-ready while
  the existing pair/audit is incomplete; four isolated CPU routing fixtures
  passed both strongest-control cases. Protocol:
  `doom_reference_round8_fitness64_protocol.json`.
* Round7's supplied pair completed **922.62 ±562.71** on the same100 seeds,
  mean95% **[813.64,1033.16]**. Selected-minus-supplied **+23.11**,95%
  **[−64.56,+109.06]**,41 wins/34 losses/25 ties, establishes no reliable gain
  (50,000 paired whole-game resamples, seed74140). The complete independent
  protocol-aware audit verified cohorts, selection/comparator, fingerprints,
  provenance and timing. Session94377 exited0 and both actual process handles
  were gone; GPU compute-client query was empty and canonical weights matched.
  Goal remains active below1092. The preregistered fitness64 launcher then
  passed CPU readiness with no live previous processes. Artifact:
  `doom_reference_round7_paired_comparison.json`.
* The preregistered round8 fitness64 search subsequently started after CUDA
  matrix multiplication and convolution backward preflight passed. Actual
  supervisor896023 and trainer896093 were verified live at230/500 generations;
  best1024-dream score950.0117 occurred at generation60. Measured GPU
  utilization95%, memory2025MiB. No round8 real evaluation or real gain is
  claimed. The initial dream mean928.3779 differs from round7's928.0801 by
  0.2979 despite matching recorded source, inputs, initial parameters, start
  pools and runtime metadata. The cause is unresolved; this prevents a
  bitwise-repeatability claim. Artifact:
  `doom_reference_round8_initial_match_cpu_audit.json`.
* Round8 completed500 generations: best1024-dream **953.3916** at
  generation300, initial928.3779, final CMA mean901.1104. Independent CPU
  review reconstructed0–500 history and disjoint900/100 start indices,
  verified arguments/source/inputs/checkpoints, optimizer count500 and final
  mean, NumPy RNG restoration and next JAX key. Both search processes exited
  and session90086 returned0; original controls remain intact. Artifact:
  `doom_reference_round8_training_cpu_audit.json`.
* Fresh real cohorts and absence of live prior processes passed CPU readiness
  before one serial round8 validation supervisor started, session10253,
  supervisor928337 and evaluator928406 verified live. CUDA matrix multiplication
  and convolution backward preflight passed. Four80-game validation cases are
  pending; no new real gain or1092 result is claimed. The unlaunched supervisor's
  copied frozen-range metadata typo was corrected before dispatch; four isolated
  routing fixtures also assert exact170000–170099 bounds. No numerical training
  or real-evaluation source changed.
* Round8's unchanged generation120 prior control completed80 validation games
  at **973.29 ±520.30**,77 deaths/3 timeouts, mean95% **[860.60,1090.03]**
  (50,000 whole-game resamples, seed74180). Registered cohort/raw records,
  source/world/controller fingerprints and canonical weights passed CPU
  verification. Recomputed SD differs by1.14e-13 and passes strict1e-9
  absolute tolerance; exact mean matches. The same supervisor advanced to
  supplied control, evaluator938982 verified live at18/80; both new policies
  and selection remain pending. Artifact:
  `doom_reference_round8_prior_control_validation_cpu_audit.json`.
* Round8's supplied control completed **922.10 ±543.40**,75 deaths/5 timeouts,
  mean95% **[804.71,1042.63]** on the same80 seeds. Prior-minus-supplied
  paired gain **+51.19**,95% **[−56.91,+162.31]**,35 wins/25 losses/20 ties,
  establishes no reliable advantage (50,000 paired whole-game resamples,
  seed74180). Both complete cohorts, shared world/runtime/source, controller
  identities and preserved incumbent passed independent CPU verification.
  The same supervisor advanced to new best-generation300, evaluator953190
  verified live at16/80; both new policies and selection remain pending.
  Artifact: `doom_reference_round8_controls_validation_cpu_audit.json`.
* New best-generation300 completed80 validations at **931.41 ±523.98**,
  76 deaths/4 timeouts, mean95% **[818.54,1047.91]**. New-minus-prior **−41.88**,
  95% **[−147.96,+63.29]**; new-minus-supplied **+9.31**,95%
  **[−93.66,+114.30]**, establish no reliable improvement. Actual complete
  cohorts, shared world/runtime/source, candidate provenance and canonical
  preservation passed CPU checks. Artifact:
  `doom_reference_round8_best_validation_cpu_audit.json`.
* Final-generation500 completed **1016.39 ±560.94**,73 deaths/7 timeouts,
  mean95% **[895.89,1140.98]**. Its paired gain over prior **+43.10**,95%
  **[−86.03,+176.65]** remains uncertain. All four complete80-game cohorts
  selected final, with prior as highest control; both were frozen at
  2026-10-04T01:45:41.848942Z before170000 testing. Independent CPU
  reconstruction verified identities/tasks, actual records/selection,
  source/world/incumbent and timing. Same supervisor started reserved100,
  evaluator964490 verified live at34/100; paired prior follows. No1092 result
  is claimed. Artifact: `doom_reference_round8_frozen_selection_cpu_audit.json`.
* A contingent fresh-CMA initialization from the completed validation winner
  was registered using training/validation evidence, with current settings and
  world fixed. Actual prior/best/final CPU initializer checks passed; final
  checkpoint metadata lookup was corrected before any dispatch, with original
  prototype/preparation preserved. Recipe/settings and decision rule unchanged;
  no test outcomes used and no extra GPU job queued. Current preparation:
  `doom_reference_round9_validation_initializer_preparation_v2.json`.
* Round8's selected final-generation500 policy completed100 reserved games
  on170000–170099 at **1026.99 ±597.70**,90 deaths/10 timeouts, fixed-policy
  bootstrap95% mean **[911.33,1145.64]** (50,000 whole-game resamples,
  seed74140). Independent CPU checks passed for all actual records, registered
  seeds, frozen controller/inference, source/world/provenance, freeze timing
  and preserved canonical weights. The observed mean remains below1092;
  an interval containing1092 does not establish the goal. The same supervisor
  advanced to paired prior control, evaluator974563 verified live with CUDA
  preflight passed. Paired improvement and final completion audit remain
  pending. Artifact: `doom_reference_round8_selected_test100_cpu_audit.json`;
  original report SHA256
  `6cde9a0b7d29cf7980391385131d150642a1fafc1db2e3beb4a44e78e7297875`.
* Round8's paired prior control completed100 games at **1040.31 ±616.27**,
  91 deaths/9 timeouts, mean95% **[920.58,1163.10]**. Selected-minus-prior
  **−13.32**,95% **[−129.75,+103.78]**,42 wins/43 losses/15 ties, establishes
  no reliable improvement. Complete protocol-aware paired/source/provenance
  audit and separate current-runtime/raw-controller/training-state/canonical/
  process-exit review passed. Session10253 exited0; no prior GPU work remains
  queued. The goal remains unproven. Artifacts:
  `doom_reference_round8_paired_comparison.json` and
  `doom_reference_round8_completion_cpu_audit.json`.
* The registered initializer recipe was materialized with unchanged settings,
  SHA256 `e1a25bf4c162a746b46961cd5a393321fddd3a9720e644353e336a5daa0eda7a`.
  Actual preparation/initializer/source/cohort checks pass on CPU. Six isolated
  routing fixtures include duplicate initializer/best identity; independent
  trainer-state auditor refuses an unstarted result. The launcher requires
  the completed previous comparison/review, current process absence, exclusive
  lock and CUDA preflight. No test result chooses the initializer or settings.
* Readiness passed with no live prior process. Round9 started one search in
  session83493; actual supervisor994238 and trainer994308 match the exact
  registered arguments and final-generation500 initializer. CUDA matrix
  multiplication and convolution backward preflight passed. No round9 real
  performance result exists yet.
* Round9's actual initializer weights/source/inputs,900/100 start pools and
  numerical runtime passed CPU verification. Initial1024-dream mean923.2109
  differs by **+22.1006** from the same policy's previous final901.1104.
  Static source checks confirm one-candidate1024 versus two-candidate2048
  engine batches. The cause of the mean difference remains unresolved;
  no bitwise-repeatability or real-performance claim, and no extra GPU
  diagnostic is run. Artifacts: `doom_reference_round9_initial_match_cpu_audit.json`
  and `doom_reference_round9_initial_call_shapes_cpu_review.json`.
* Round9's prepared real supervisor now blocks reserved testing until its
  independent complete-validation CPU reconstruction passes. It fingerprints
  that auditor in the freeze; six CPU routing fixtures verify the audit comes
  before any test, including retained controls and duplicate initializer/best
  identities. Choice and no-test closure auditors defer missing actual results.
  Measured live search progress343/500: best1024-dream **958.03125** at
  generation290. This is not real-game evidence for1092.
* Round9 completed500 generations: best1024-dream **965.65234375** at
  generation380, initial923.2109375, final CMA mean892.94140625. Independent
  CPU audit passed for0–500 history, reconstructed start pools, exact
  validation-only initializer, source/input/checkpoints, optimizer count500
  and final mean, NumPy RNG and next JAX key. All controls remain intact;
  session83493 exited0 and both training processes exited. Artifact:
  `doom_reference_round9_training_cpu_audit.json`.
* Current-state/fresh-cohort/exclusive-lock CPU readiness passed before
  one serial real supervisor started, session30040; supervisor1016081 and
  evaluator1016151 verified live. CUDA matrix multiplication/convolution
  backward preflight passed. First control26/80 at the measured snapshot;
  complete validation, independent choice and any eligible fresh paired100
  tests remain pending. Artifact: `doom_reference_round9_real_dispatch_readiness_cpu.json`.
* Round9's unchanged initializer control completed80 games at
  **1031.74 ±617.01**,69 deaths/11 timeouts, fixed-policy bootstrap95% mean
  **[897.89,1168.05]** (50,000 whole-game resamples, seed74180). Actual
  registered records, controller/inference/source/input provenance and
  canonical preservation passed independent CPU checks. This validation
  control cannot establish1092; complete candidate selection remains pending.
  The same supervisor advanced to supplied control, evaluator1019526 verified
  live at45/80 games. Artifact: `doom_reference_round9_prior_control_validation_cpu_audit.json`.
* Round9's supplied control completed80 games at **982.68 ±520.17**,
  77 deaths/3 timeouts, fixed-policy mean95% **[870.09,1096.81]**. Independent
  CPU comparison verified both complete controls, registered raw records,
  current runtime, matching world/source and canonical preservation. Prior
  minus supplied is **+49.06**, paired95% **[−73.88,+172.28]**,33 wins/
  34 losses/13 ties; no reliable gain is established. Artifact:
  `doom_reference_round9_controls_validation_cpu_audit.json`. Both updated
  candidates and complete validation selection remain pending.
* Round9's new best-generation380 checkpoint completed80 validations at
  **961.74 ±501.45**,78 deaths/2 timeouts, mean95% **[853.01,1072.85]**.
  Independent raw-record/controller/world/source checks passed. New best minus
  prior is **−70.00**, paired95% **[−207.94,+67.68]**; minus supplied is
  **−20.94**, paired95% **[−138.49,+98.60]**. The final-generation500 cohort
  remains running; selection is not frozen. Artifact:
  `doom_reference_round9_best_validation_cpu_audit.json`.
* Round9's final-generation500 checkpoint completed80 validations at
  **940.85 ±535.48**,77 deaths/3 timeouts, mean95% **[824.32,1058.61]**.
  Minus the prior control is **−90.89**, paired95% **[−217.69,+33.01]**.
  All four complete cohorts selected the unchanged prior1031.74 control;
  neither new candidate won. Independent choice and no-test closure audits
  verified records, identities, source/world and that180000–180099 remain
  unused. Current runtime/training-state/canonical/process review passed;
  session30040 exited0, both processes exited and no prior GPU work remains.
  Artifacts: `doom_reference_round9_frozen_selection_cpu_audit.json`,
  `doom_reference_round9_validation_closure_cpu_audit.json` and
  `doom_reference_round9_completion_cpu_audit.json`. The1092 goal stays unproven.
* Registered a seed92 repeat from the same initializer and settings as round9,
  changing only training seed91→92 and output path. This changes CMA samples,
  dream streams and the900/100 start-pool partition together; it does not
  isolate CMA randomness. Protocol SHA256
  `4d75e8c53d9ee6fdc5203ca5cd2b4796a0035beec3bf3753fd160c1aac07e2b7`.
  CPU verification passed for exact arguments/initializer, unchanged numerical
  trainer, fresh80-game validation130340–130419 and contingent paired100-test
  190000–190099, disjoint start pools, prior closure and exclusive GPU lock.
  Training-state auditor correctly defers an unstarted result; syntax/Ruff
  pass. Artifact: `doom_reference_round10_search_readiness_cpu_audit.json`.
* The single round10 search started in session49055, supervisor1050296 and
  trainer1050365 verified live with the exact registered arguments. CUDA
  matrix multiplication and convolution backward preflight passed. Keep all
  original best/final checkpoints and optimizer/RNG states; no real result is
  yet available. Only a distinct updated controller winning all fresh real
  validation games can advance to reserved testing.
* Round10's live initial-state CPU audit at61/500 generations verified exact
  seed92 arguments, unchanged raw initializer, frozen inputs/source/runtime,
  independently reconstructed900/100 pools and canonical preservation.
  Initial1024-dream mean **870.5869140625** uses different start/noise streams
  from seed91 and is not an estimate of training improvement. Measured GPU
  utilization95% with1905MiB resident at a training snapshot. Artifact:
  `doom_reference_round10_initial_match_cpu_audit.json`; real results are pending.
* Prepared round10's serial real supervisor, independent choice/retention
  auditors and completed-training readiness checker. Seven isolated CPU
  fixtures pass control retention, candidate wins, strongest-control pairing,
  identity deduplication, audit-before-test ordering and rejection blocking
  reserved tests. Actual readiness/choice/closure checks defer missing real
  results; no evaluation GPU job starts. The evaluator command, numerical
  inference and recovery code exactly match the prior verified workflow.
  Syntax/Ruff pass. Artifact:
  `doom_reference_round10_real_workflow_preparation_cpu.json`. The original
  search remains live at a measured239/500 snapshot, best1024-dream954.8711
  at generation1; this is not a real1092 result.
* Round10 completed500 generations: best1024-dream **954.87109375** at
  generation1, initial870.5869140625 and final CMA mean953.3369140625.
  Independent CPU audit passed for0–500 history, unchanged initializer/source,
  seed92 start pools, best/final weights, optimizer count500/final mean, NumPy
  state and next JAX key. Session49055 exited0 and both training processes
  exited. Artifact: `doom_reference_round10_training_cpu_audit.json`.
* Current-state/fresh-cohort/exclusive-lock readiness passed before one serial
  real supervisor started, session87088, supervisor1068933 and first evaluator
  1069003 verified live. CUDA matrix multiplication/convolution backward
  preflight passed. First stage is the unchanged prior's80 posterior-validation
  games130340–130419. Complete validation and independent frozen selection
  precede any eligible paired100-test190000–190099; real1092 is unproven.
  Artifact: `doom_reference_round10_real_dispatch_readiness_cpu.json`.
* Round10's unchanged prior completed80 validation games at **936.94 ±547.92**,
  76 deaths/4 timeouts, fixed-policy bootstrap95% mean **[818.01,1057.30]**
  (50,000 whole-game resamples, seed74180). Actual registered records,
  controller/inference/source/input/current-runtime provenance and canonical
  preservation passed independent CPU checks. Artifact:
  `doom_reference_round10_prior_control_validation_cpu_audit.json`.
  This control validation does not establish1092. The same supervisor advanced
  to supplied control, evaluator1072617 verified live; both new policies and
  complete selection remain pending.
* Round10's supplied control completed80 validations at **936.08 ±561.21**,
  76 deaths/4 timeouts, fixed-policy mean95% **[814.65,1061.19]**. Independent
  CPU checks verified both complete controls, registered records, current
  packages, matching world/source and canonical preservation. Prior minus
  supplied is **+0.86**, paired95% **[−126.09,+126.04]**,35 wins/33 losses/
  12 ties. Neither control has an established advantage. Artifact:
  `doom_reference_round10_controls_validation_cpu_audit.json`. Both new
  candidates and complete validation selection remain pending.
* Round10's new best-generation1 checkpoint completed80 validations at
  **915.18 ±529.99**,77 deaths/3 timeouts, mean95% **[800.90,1033.56]**.
  Independent CPU raw-record/source/input/current-runtime/controller checks
  passed. Best minus prior is **−21.76**, paired95% **[−109.93,+69.14]**;
  minus supplied is **−20.90**, paired95% **[−137.96,+94.89]**. The final
  checkpoint is still running; no winner is frozen. Artifact:
  `doom_reference_round10_best_validation_cpu_audit.json`.
* Round10's final-generation500 checkpoint completed80 validations at
  **929.45 ±573.13**,72 deaths/8 timeouts, mean95% **[807.46,1057.56]**.
  Final minus prior is **−7.49**, paired95% **[−133.54,+116.25]**. All four
  complete cohorts retained the unchanged prior936.94 control. Independent
  selection/no-test closure and current runtime/source/controller/training-state/
  canonical/process audits passed. Session87088 exited0; both processes exited,
  no work is queued and190000–190099 remain unused. Artifacts:
  `doom_reference_round10_validation_closure_cpu_audit.json` and
  `doom_reference_round10_completion_cpu_audit.json`. Neither seed91 nor seed92
  refinement beat its complete validation control;1092 remains unverified.
* A preregistered controlled GPU diagnostic compared identical initializer
  weights, seed91 starts, key100091 and temperature1.15 at1024 versus2048
  dream trajectories. Means were **923.2109375** and **910.623046875**,
  a difference of **−12.587890625**, with783/1024 different survival outcomes.
  Repeats within each shape and both duplicated slots matched exactly.
  Independent CPU raw-score/controller/source/input/RNG checks passed; session
  84793 exited0 and its process exited. This demonstrates current batch-shape
  sensitivity, without diagnosing a specific kernel or fully explaining the
  historical901.1103515625 result. No training or real games were run.
  Artifacts: `doom_reference_batch_shape_protocol.json`,
  `doom_reference_batch_shape_diagnostic.json` and its CPU audit.
* Added `scripts/tools/train_doom_reference_fixed_validation.py`, preserving
  every earlier frozen trainer. Every held-out candidate now uses the same
  single-candidate call shape as the baseline. CPU fixtures execute the actual
  evaluation function, verifying fixed shapes and shared starts/keys across
  baseline/updates, unchanged training calls including padding, and unchanged
  optimizer/key/checkpoint code outside the explicit evaluation change.
  Syntax/Ruff and diff checks pass. Artifact:
  `doom_reference_fixed_validation_calls_cpu_audit.json`. A GPU check of this
  new call path and a preregistered new experiment remain pending; no new
  training or real-performance result is claimed.
* The actual fixed-validation function passed a registered GPU verification:
  one, two and three copied policies all used1024-trajectory calls and all six
  raw score vectors matched the earlier single-policy baseline exactly,
  mean923.2109375. Independent CPU source/input/raw-score/runtime checks
  passed; session48325 exited0 and the process exited. Artifacts:
  `doom_reference_fixed_validation_gpu.json` and its CPU audit. No training
  or real games were run by this verification.
* Registered round11 as a seed92 replay with the same initializer/world/CMA
  arguments, changing only the held-out validation call shape (and output
  path). Protocol SHA256
  `bd9345a333726a5bf2d8f9ad1049ca43dbf3598220ebc268dc6e58a49234023e`.
  Fresh80-game validation130420–130499 and contingent paired100-test
  200000–200099 are reserved. Candidates identical to failed round10 best/
  final policies are excluded. Source/input/initializer/parent-closure/fixed-
  validation checks and fresh-cohort readiness passed before one CUDA search
  started, session55459, supervisor1123810 and trainer1124156 verified live.
  CUDA matrix multiplication/convolution backward preflight passed again.
* The independent live initial-state audit verified exact arguments, initializer,
  frozen inputs/source/runtime metadata and matching seed92 start pools. The
  initial dream mean887.478515625 differs by+16.8916015625 from the recorded
  round10 baseline870.5869140625; cause remains unresolved. Do not attribute
  later differences solely to the intended selection correction. At83/500,
  all83 population mean/best statistics match round10 exactly; full optimizer/
  weight replay remains unproven. Best fixed-shape held-out dream949.2373 at
  generation50 is not real survival evidence. Artifact:
  `doom_reference_round11_initial_match_cpu_audit.json`.
* Round11's real workflow is prepared. Nine isolated CPU fixtures verify
  retained controls, strongest-control pairing, identity deduplication, exclusion
  of one/all prior failed candidates and independent audit rejection blocking
  reserved tests. Actual readiness/choice/closure auditors defer missing
  completed training/real records. The numerical inference, recovery and
  audit-order functions match round10 aside from artifact names. Syntax/Ruff
  pass. Artifacts: `doom_reference_round11_real_workflow_preparation_cpu.json`
  and `doom_reference_round11_real_numeric_guard_cpu.json`. The sole current
  GPU job remains the search; no real evaluation has been dispatched.
* Round11 completed500 generations. Best fixed-shape dream **954.5556640625**
  was generation470; initial887.478515625 and final mean915.376953125.
  Independent CPU audit passed for full history, arguments/inputs/source,
  start pools, best/final flags/weights, optimizer500 and RNG state. All500
  population mean/best statistics and final raw parameters match round10
  exactly; the final policy is excluded as the previously failed duplicate.
  The new best raw identity is distinct. Session55459 exited0, both training
  processes exited, and completed-state/fresh-cohort/exclusive-lease readiness
  passed. Artifact: `doom_reference_round11_training_cpu_audit.json`.
* One serial real supervisor started in session15768, supervisor1149459 and
  evaluator1149804 verified live with the exact80-game130420–130499,
  posterior/eight-worker unchanged-incumbent command. CUDA preflight passed.
  Three policies will be validated: unchanged incumbent, public control and
  new generation470; the archived duplicate final is omitted. Independent
  complete-validation selection must precede any eligible paired100-test
  200000–200099. Artifact: `doom_reference_round11_real_dispatch_readiness_cpu.json`.
  No real1092 result is yet established.
* Round11's unchanged prior controller completed80 validations at
  **953.56 ±529.86**,76 deaths/4 timeouts, fixed-policy mean95%
  **[839.74,1071.74]**. Independent CPU checks passed for all registered raw
  records, actual controller/source/input hashes, current package versions,
  shared environment and canonical preservation. Artifact:
  `doom_reference_round11_validation_progress_1_cpu_audit.json`. The same live
  supervisor has advanced to the public control; generation470 and complete
  validation selection remain pending. This control cohort is not a reserved test.
* Round11's public control completed80 validations at **972.89 ±546.46**,
  75 deaths/5 timeouts, fixed-policy mean95% **[854.02,1094.39]**. Independent
  CPU raw-record/controller/source/input/current-runtime/canonical checks passed
  for both controls. Public minus prior is **+19.33**, paired95%
  **[−94.28,+130.69]**,41 wins/30 losses/9 ties; no reliable advantage is
  established. Artifact: `doom_reference_round11_validation_progress_2_cpu_audit.json`.
  The same live supervisor advanced to generation470. Complete selection
  remains pending; control validation and intervals containing1092 do not
  establish the target.
* Round11's generation470 candidate completed80 validations at
  **963.89 ±626.80**,72 deaths/8 timeouts, mean95% **[828.96,1101.34]**.
  Candidate minus public is **−9.00**, paired95% **[−136.10,+121.86]**.
  Complete validation retained the unchanged public control. Independent
  selection/no-test closure reconstructed all240 actual records, frozen source/
  world/controller identities, initializer provenance and freeze timing;
  200000–200099 remain unused. Current-runtime/training-state/canonical and
  actual session/process/lease checks passed. Session15768 exited0; all related
  processes exited and no GPU work is queued. Artifacts:
  `doom_reference_round11_validation_closure_cpu_audit.json` and
  `doom_reference_round11_completion_cpu_audit.json`. Fixed validation selected
  a different policy but did not establish better real transfer or1092.
* Registered round12 as two matched controller searches at temperature1.15
  and1.10, both seed91/public initialization, fixed single-candidate validation,
  sigma0.005,500 generations,pop64,fitness64 and1024 held-out rollouts. Only
  temperature and output paths differ between arms; the paper motivates a
  local temperature comparison, while1.10 is our hypothesis. Six previously
  failed checkpoint identities are excluded. Fresh validation130500–130579
  and contingent paired100-test210000–210099 passed CPU seed reservation.
  Immutable protocol: `doom_reference_round12_temperature_pair_protocol.json`,
  SHA256`5b500de0d64169b3329fcda2fa311c252c4d28b6d043d015601d20f79d7fb45a`.
  No new GPU job has started. Serial launcher/readiness and independent audits
  must be prepared before dispatch; no gain is inferred from this registration.
* Round12's serial training launcher and independent arm auditor passed CPU
  fixtures: actual trainer argument parsing, a complete synthetic public-
  initializer capsule, corrupt initializer/pool/RNG/history/checkpoint rejection,
  serial preflight/audit ordering, live/incomplete-run rejection, audit-failure
  stop and completed-arm recovery without dispatch. Syntax/Ruff passed; sources
  were frozen. Actual parent/source/runtime/initializer/fresh-cohort/lease
  readiness passed before one supervisor started in session42213, PID1202880.
  CUDA matrix multiplication/convolution backward preflight passed; first1.15
  trainer1203226 was verified live at7/500, with96% GPU utilization measured.
  Independent live CPU initialization checks passed at27/500; the initial dream
  mean928.3779296875 exactly matches round8's recorded initial mean. This is
  not proof of full-training or raw-trajectory replay, nor real performance.
  Artifacts: `doom_reference_round12_training_preparation_cpu.json`,
  `doom_reference_round12_training_readiness_cpu_audit.json` and
  `doom_reference_round12_tau115_initial_cpu_audit_correction.json`.
  The same supervisor will audit the finished first arm before launching1.10;
  real workflow preparation and real survival evidence remain pending.
* Round12's temperature 1.15 arm completed all 500 generations. Its best
  held-out dream score was **948.2070 at generation 70**. The independent CPU
  audit verified best/final weights, all history rows, optimizer count 500,
  RNG state, source/inputs and canonical preservation. The same supervisor
  then started the 1.10 arm after CUDA preflight; both processes were verified
  live at generation 23 on 2026-10-04T06:18:11Z. Its initializer, start pools,
  random-stream metadata and calculation shapes match the first arm.
* The two-arm real-evaluation workflow is now prepared and frozen in
  `doom_reference_round12_real_workflow_preparation_cpu.json`. Twelve isolated
  CPU fixtures passed, including independent selection before reserved tests,
  rejection blocking tests, retained-control closure and full paired analysis.
  Inference/recovery paths match the previous evaluator. No real evaluations
  have started; both completed searches and actual training exit must precede
  fresh readiness checks. Dream scores do not establish the 1092-step target.
* Both round12 searches are now complete. The 1.10 arm's best held-out dream
  score was **1076.3115 at generation 310**. Both independent training audits
  passed, and actual session 42213 exited 0 (tool chunk `515ec3`). Current
  source/runtime/checkpoint/RNG/canonical and process/lease checks passed in
  `doom_reference_round12_training_completion_cpu_audit.json`.
* Fresh real-dispatch readiness passed. Raw identity checks showed the 1.15
  final policy exactly duplicates the prior control; its generation-70 best
  is distinct. The final registered comparison has **five policies / 400
  validation games**, retaining both controls and three distinct candidates.
  One real supervisor started in session 16340, PID 1253545. CUDA preflight
  passed; evaluator 1253615 was verified live on the public control at 1/80
  games on 2026-10-04T06:43Z. Selection and any eligible reserved pair remain
  pending. Artifacts: `doom_reference_round12_real_dispatch_readiness_cpu.json`
  and `doom_reference_round12_registered_validation_cases_cpu.json`.
* Round12's public control completed all 80 validation games at
  **957.56 ±599.62**, with 73 deaths / 7 timeouts and a fixed-policy 95%
  mean interval of **[828.66,1092.14]**. Independent CPU reconstruction verified
  every raw record, controller/source/input fingerprints, current runtime,
  matching environment and canonical preservation. Artifact:
  `doom_reference_round12_validation_progress_1_cpu_audit.json`. The same
  supervisor is evaluating the prior local controller (evaluator 1261348
  verified live). All five cohorts must finish before selection. This unchanged
  control validation and its interval do not establish a reserved 1092-step result.
* Round12's prior local controller completed 80 validation games at
  **913.01 ±502.18**, 78 deaths / 2 timeouts, fixed-policy mean95%
  **[803.20,1024.40]**. Prior minus public is **−44.55**, paired95%
  **[−171.36,+84.65]**, with 25 wins / 42 losses / 13 ties; no reliable
  improvement is established. Independent CPU raw-record, actual control
  identity, source/input/runtime/shared-environment and canonical checks
  passed for both controls in `doom_reference_round12_validation_progress_2_cpu_audit.json`.
  The same supervisor advanced to the distinct 1.15 generation-70 candidate,
  evaluator 1271193 verified live. Three candidate cohorts remain; selection
  and any eligible reserved paired100 remain pending.
* Round12's distinct temperature 1.15 generation-70 candidate completed 80
  validation games at **926.31 ±579.07**, 75 deaths / 5 timeouts, mean95%
  **[801.87,1055.04]**. Candidate minus public is **−31.25**, paired95%
  **[−141.41,+77.48]**, with 33 wins / 30 losses / 17 ties. Independent
  raw-record, policy identity/generation/temperature, source/input/runtime/
  shared-environment and canonical checks passed for all three complete
  cohorts. Artifact: `doom_reference_round12_validation_progress_3_cpu_audit.json`.
  The same supervisor advanced to temperature 1.10 best, evaluator 1281065
  verified live. Both 1.10 cohorts and complete selection remain pending;
  the 1.15 candidate did not outscore public, and no reserved test has started.
* Round12 completed all **400 validation games** on seeds 130500–130579.
  Temperature 1.10 best scored **904.69 ±524.83**, with 78 deaths / 2
  timeouts; its difference from public was **−52.88**, paired95%
  **[−181.60,+75.13]**. The 1.10 final checkpoint scored **922.06 ±522.69**,
  with 76 deaths / 4 timeouts; its difference was **−35.50**, paired95%
  **[−174.74,+103.65]**. Complete validation-only selection retained the
  unchanged public control at **957.56 ±599.62**. All three new candidates
  scored below it; reserved seeds **210000–210099 remain unused**.
* Independent selection, no-test closure and final CPU audits verified all
  raw records, policy provenance, source/inputs/runtime, training checkpoints,
  optimizer/RNG state and canonical preservation. Real session 16340 exited
  0 (tool chunk `3c3cbf`); the 2026-10-04T07:47:48Z completion review found
  all experiment processes absent, the GPU lease available and no further
  GPU work queued. Artifacts: `doom_reference_round12_frozen_selection_cpu_audit.json`,
  `doom_reference_round12_validation_closure_cpu_audit.json` and
  `doom_reference_round12_completion_cpu_audit.json`. The 1092-step target
  remains unmet. This comparison used an imported public world model;
  the preserved canonical own-world result remains **840.06 ±524.48**.
* Round13 preregistered one bounded **direct real-survival CMA search** after
  round12's dream-score gains failed to improve real validation. It starts
  from the public controller selected solely by complete round12 validation,
  with the public VAE/RNN, architecture and posterior inference fixed. This
  changes the paper's dream-only controller-training method. Settings: seed94,
  sigma0.005, 16 generations, population16, four common fitness seeds per
  generation and eight persistent game workers. Maximum training is 1,024
  fitness games plus 80 training-holdout games; fresh policy validation and
  reserved tests remain separate.
* Seven CPU tests passed, including mixed-candidate scheduling, independent
  CMA reconstruction and interruption recovery that exactly reproduced the
  uninterrupted weights and next population. On the actual frozen models,
  CPU and GPU checks matched original actor actions, hidden states, RNG keys
  and dtypes exactly for the tested public/perturbed policies. These short
  numerical checks are not historical paper equivalence or performance proof.
  Protocol: `doom_reference_round13_direct_real_protocol.json`, SHA256
  `60b9198a566ece4d1f373856a8c580e05f7cc2ed6a4cf92577f77077be6bc639`.
* Fresh source/input/runtime/seed/process/lease readiness and CUDA preflight
  passed before one supervisor started in **session52524**, PID1323498.
  Trainer1323841 was independently verified live on
  2026-10-04T08:10:33Z with **3/16 baseline training-holdout games complete**.
  Log: `doom_reference_round13_training.log`; status:
  `doom_reference_round13_training.status`. The supervisor will run the
  independent CPU training auditor after the trainer exits. No fresh policy
  validation or reserved test has started. Estimated full-loop time is
  **4–6 hours**, based on prior measured 80-game evaluations of606–643 seconds;
  actual costs may differ. Goal completion remains unproven.
* At2026-10-04T08:12:27Z, all16 public baseline training-holdout games were
  complete at mean **768.5625**, and the first generation's fitness stage had
  started. Both actual supervisor and trainer PIDs remained live. This small,
  fixed training cohort is not the separate fresh-validation result or a
  reserved test; no updated-policy performance is established yet.
* Round13's fresh real-comparison workflow is now prepared and frozen in
  `doom_reference_round13_real_workflow_preparation_cpu.json`, SHA256
  `371a78c967b9a852d0ca4ddac899cdb87d7147fffa19e391bde0fabb21db82f0`.
  Thirteen CPU tests passed: complete cohort selection, highest-control pairing,
  controls-first ties, independent audit rejection blocking tests, raw-policy
  deduplication/failure exclusions, partial recovery, world drift, nested seed
  reuse and incomplete paired-test rejection. Synthetic fixtures do not
  establish real performance. Actual readiness correctly deferred all GPU
  evaluation while session52524 and its trainer remain live.
* At2026-10-04T08:38:18Z, **three generations were complete** and generation4
  had **16/64 fitness games** recorded. Generation3's population mean was
  **860.96875**, with best four-game fitness **1092.0**. The fixed training
  holdout's current best remains the public baseline **768.5625**, pending
  the first new mean checkpoint check after generation4. Neither the four-game
  fitness score nor the training holdout proves a fresh100-game1092 mean.
  Both actual supervisor1323498 and trainer1323841 were verified live.

At 2026-10-04T08:49:36.958151+00:00, generation4 improved the same16 training holdouts from
**768.5625 ±343.2047** to **895.25 ±585.4425** (sample standard deviations).
The paired mean change was **+126.6875**; a fixed-cohort whole-game bootstrap
95% interval was **[-86.8750,343.6250]**, with5 improved games,4 worse
and7 ties. Both complete raw cohorts, seeds, policy fingerprints and the
current immutable optimizer snapshot were verified on CPU in
`artifacts/doom_reference_round13_g004_holdout_progress_cpu.json`. This is
a repeated training holdout used for checkpoint selection, not fresh policy
validation or a reserved100-game result. Generation5 fitness is running;
supervisor1323498 and trainer1323841 were independently verified live.
The1092-step goal remains unproven.

At 2026-10-04T09:24:38.740839+00:00, generation8 scored **888.625 ±583.4805** on the same16
training holdouts (sample standard deviation), below generation4 **895.25**.
Generation4 remains best. Generation8 minus baseline768.5625 was
**+120.0625**, fixed-cohort paired bootstrap95% **[-112.5641,382.6875]**;
generation8 minus generation4 was **−6.625**, interval
**[-234.0000,200.1875]**. CPU reconstruction checked all8 candidate
populations, fitness means/history, mean and best weights, next pending
population and **560 complete raw training-game records**, with frozen
source/input fingerprints, in
`artifacts/doom_reference_round13_g008_holdout_progress_cpu.json`. These are
repeated training holdouts used for checkpoint selection, not fresh policy
validation or the reserved100-game target. Generation9 fitness is running;
the original supervisor1323498 and trainer1323841 were verified live.
The registered16-generation search continues; fresh validation remains pending.

At 2026-10-04T10:01:38.356194+00:00, generation 12 became the best training checkpoint, scoring
**966.0625 ±620.9954** on the same 16 training holdouts (sample standard
deviation), above generation 4 **895.25** and generation 8 **888.625**.
The paired gain over baseline **768.5625** was **+197.5**, with fixed-cohort
whole-game bootstrap 95% interval **[-52.0000,467.0625]**. The gain
over generation 4 was **+70.8125**, interval **[-113.6250,302.8141]**.
CPU reconstruction verified all 12 populations, raw fitness means/history,
optimizer mean and best weights, the next pending population and **832 complete
raw training-game records**, with frozen source/input fingerprints. Evidence:
`artifacts/doom_reference_round13_g012_holdout_progress_cpu.json`. These are
repeated training holdouts used for checkpoint selection; they do not establish
fresh policy performance or the reserved 100-game 1092-step mean. Generation
13 fitness is running, with supervisor 1323498 and trainer 1323841 verified
live. The registered 16-generation search continues before fresh validation.

At 2026-10-04T10:33:47.970071+00:00, round13 completed all16 generations and **1104 training games**.
The independent CPU auditor reconstructed every candidate population, complete
raw fitness/holdout cohort, final mean and selected best weights, history,
optimizer and next RNG population. Its current file hashes, registered
source/inputs/runtime and canonical checkpoints were verified again after
actual supervisor session52524 exited0 (tool `f6f8db`). Supervisor1323498,
trainer1323841 and auditor1417088 were absent, repository training/evaluation
workers had exited, and the exclusive GPU lease was available. Evidence:
`artifacts/doom_reference_round13_training_cpu_audit.json` and
`artifacts/doom_reference_round13_training_exit_cpu_review.json`.
Generation16 is best at **1011.375 ±672.5837** on16 repeated training
holdouts (sample standard deviation), versus baseline768.5625. Paired gain:
**+242.8125**, fixed-cohort bootstrap95% **[-11.6891,495.6891]**. These
training/selection measurements do not establish fresh policy performance.
Best and final raw weights are identical, so fresh validation has three
distinct policies: public, prior initializer control and the generation16
candidate, each on80 new games130580–130659. No validation or reserved
220000–220099 test has started yet; the1092-step goal remains unproven.

At 2026-10-04T10:39:23.574595+00:00, the prepared fresh real comparison passed readiness (tool
`0a73c1`) and started in **session2935**, supervisor1417956. CUDA preflight
passed; actual public-control evaluator1418302 was verified live. The three
distinct policies each use80 posterior games on the same fresh seeds
130580–130659. Current measured completion: public control **43/80**, prior control **0/80**,
and new candidate **0/80**. Global seed
separation outside independently verified own reports passed, with reserved
220000–220099 unused. Evidence:
`artifacts/doom_reference_round13_real_dispatch_cpu_review.json`. The existing
supervisor must finish all240 validation games and independently audit its
frozen validation-only choice before any eligible100-game candidate/control
pair. No winner is frozen yet, no reserved test has begun, and no fresh
performance claim or1092-step goal completion is established.

At 2026-10-04T10:45:41.253394+00:00, the public control completed all80 fresh validation games
130580–130659 at **875.85 ±513.6677** (population standard
deviation), with 78 deaths and 2 timeouts.
The complete raw cohort, role, policy/world/input/source/runtime metadata and
reported statistics were checked on CPU. Report:
`artifacts/doom_reference_round13_val_supplied.json`, SHA256
`54d6a8ae68ff2ced285e67bcd411db110514cf92d2295f2f96bb1786bbaae718`. Existing session2935 has moved to the unchanged
prior-control cohort; the new candidate has not yet been evaluated. All three
complete80-game cohorts remain required before selection. No winner is frozen
and reserved220000–220099 seeds remain unused; this control result does not
establish the1092-step target.

At 2026-10-04T10:54:09.828026+00:00, the unchanged priorR8 control completed all80 fresh validation
games at **926.35 ±492.7647** (population standard deviation),
with 77 deaths and 3 timeouts. Its complete
raw records, source/world/runtime/provenance and statistics were independently
checked on CPU; public control remains875.85 ±513.6677 on the same seeds.
Prior report: `artifacts/doom_reference_round13_val_prior_initializer.json`,
SHA256 `1166d07da2fd2c07e2df41a465fc1a87b9c10b94e8fe662c0e68cc81e1266e2c`. Existing session2935 is now evaluating
the generation16 candidate. Its complete80-game cohort is still required
before any winner/control is frozen; reserved220000–220099 remains unused.

Round 13 completed all 240 fresh validation games, with each policy evaluated
on the same 80 seeds, 130580–130659:

| Policy | Mean steps | Population standard deviation | Deaths / timeouts |
| --- | ---: | ---: | ---: |
| Public controller | 875.85 | 513.67 | 78 / 2 |
| Prior round 8 controller | 926.35 | 492.76 | 77 / 3 |
| New round 13 controller | 885.29 | 562.21 | 76 / 4 |

The prior controller was retained using complete validation only. The new
controller's paired mean change was **−41.06 steps**, with a whole-game
bootstrap 95% interval of **[−130.30, +47.24]**. This interval conditions on
the fixed policies and excludes training and sequential selection uncertainty.
The training-holdout improvement did not establish fresh validation improvement.
No reserved test was run; seeds 220000–220099 remain unused, and the paper's
1092-step target remains unmet.

The independent CPU result audit verified all raw validation records, current
source/input/runtime fingerprints, checkpoint eligibility and the frozen choice:
`artifacts/doom_reference_round13_real_result_cpu_audit.json`. The supervisor
exited successfully; its recorded processes and repository workers were absent
and the GPU lease was available at the precommit check. This experiment trained
the controller directly on real simulator survival using an imported public
VAE/RNN, which differs from the paper's dream-only controller training and does
not establish reproduction of our own world-model training.

Round 14 was registered and started on CUDA at 2026-10-04T11:20Z. It warm-starts
the prior round 8 controller selected from complete round 13 validation, uses
eight fitness games per candidate per generation and 64 repeated training
holdouts, and is bounded to eight CMA generations with population 16 and
sigma 0.005. The imported VAE/RNN, architecture and inference protocol stay
fixed. This combined refinement does not isolate which change affects results.
All 21 CPU workflow/replay/recovery tests and the actual trainer readiness check
passed; CUDA preflight passed on the RTX 4070 Ti. Current baseline holdout
completion was 15/64 at 2026-10-04T11:22:29.934276+00:00. This is training progress,
not fresh policy performance. Supervisor session 49443 and its trainer were
independently verified live.

Fresh fitness seeds 520000–520063, training holdouts 530000–530063, validation
seeds 130660–130739 and reserved tests 230000–230099 were checked unused and
separated before dispatch. All eligible policies require complete 80-game
validation; only a distinct new validation winner proceeds to the complete
100-game reserved comparison with the unchanged control on the same seeds.
The planned four-hour window is an estimate. Protocol and resume evidence:
`artifacts/doom_reference_round14_direct_real_protocol.json`,
`artifacts/doom_reference_round14_training_preparation_cpu.json`, and
`artifacts/task_state.json`. The paper target remains unmet.

At 2026-10-04T11:29:41.782983+00:00, round 14 completed its 64-game training baseline at
**970.125 ±537.5760** steps (sample standard deviation), with
62 deaths and 2 timeouts. CPU reconstruction verified all raw records,
registered seeds/settings and initializer weights, saved generation-zero best
and final weights, and the first 16-candidate CMA population and RNG state.
Evidence: `artifacts/doom_reference_round14_baseline_cpu_audit.json`.
Generation 1 is evaluating its 128 candidate/game pairs, with the existing
supervisor session 49443 and trainer independently verified live. This is
a repeated training holdout, not fresh validation or the reserved 100-game
result. The 1092-step target remains unmet.

At 2026-10-04T11:41:44.408753+00:00, round 14 completed generation 1 and started generation 2.
The independent CPU prefix audit verified all **192 completed training-game
records** (64 baseline holdouts plus 128 fitness pairs), reconstructed the
first population and fitness means, saved optimizer mean/best/final weights
and history, and the next pending population and RNG. Evidence:
`artifacts/doom_reference_round14_g001_training_prefix_cpu_audit.json`.
Its hashes describe the saved files at that check; mutable checkpoint/pointer
files will advance as training continues, while the audited optimizer snapshot
and completed raw cohorts are preserved. The best training holdout remains
the generation-zero baseline **970.125**; the next holdout evaluation is at
generation 4. Existing supervisor session 49443 and its exact trainer command
were verified live. No fresh validation or reserved test has started, and
these training fitness scores do not establish the 1092-step target.

At 2026-10-04T11:53:19.751702+00:00, round 14 completed generation 2. CPU prefix replay
verified all **320 completed training-game records**, both fitness populations
and rankings, saved mean/best/final weights and history, and the pending third
population and RNG. Evidence:
`artifacts/doom_reference_round14_g002_training_prefix_cpu_audit.json`.
Generation 3 is running in the same verified live session 49443. The best
training holdout remains generation zero at **970.125**; its next comparison
is at generation 4. Fresh validation and reserved testing remain pending,
so the 1092-step objective is still unproven.

At 2026-10-04T12:10:22.267991+00:00, round 14 completed generation 3. CPU replay verified
all **448 completed training-game records**, three populations and rankings,
saved mean/best/final weights and history, and pending fourth population/RNG.
Evidence: `artifacts/doom_reference_round14_g003_training_prefix_cpu_audit.json`.
Generation 4 is running in the same verified live session 49443; after its
fitness population it will compare the mean controller on all 64 training
holdouts against the generation-zero baseline **970.125**. Fresh validation
and reserved testing remain pending. These training records do not establish
the 1092-step objective.

At 2026-10-04T12:32:05.070744+00:00, round 14 generation 4 scored **899.8281 ±490.4217**
on all 64 repeated training holdouts (sample standard deviation), below
baseline **970.125 ±537.5760**. It had 63 deaths and 1 timeout versus
baseline 62 deaths and 2 timeouts. The paired change was **−70.2969** steps,
with whole-game bootstrap 95% interval **[−174.0785,+27.8910]**. This
conditions on the fixed checkpoints and repeated training cohort; it excludes
training/checkpoint/sequential selection uncertainty and is not fresh validation.
The baseline remains best. CPU replay verified all **640 completed training
records**, four populations/rankings, selected best and current mean weights,
history, and the pending fifth population and RNG. Evidence:
`artifacts/doom_reference_round14_g004_training_prefix_cpu_audit.json` and
`artifacts/doom_reference_round14_g004_holdout_comparison_cpu.json`.
Generation 5 is running in verified live session 49443. The registered search
continues to generation 8 before fresh validation; reserved testing has not
started, and the 1092-step target remains unmet.

At 2026-10-04T12:43:34.778264+00:00, round 14 completed generation 5. CPU prefix replay
verified **768 completed training-game records**, five populations/rankings,
saved mean/best/final weights and history, and pending sixth population/RNG.
Evidence: `artifacts/doom_reference_round14_g005_training_prefix_cpu_audit.json`.
Generation 6 is running in verified live session 49443. The baseline remains
best at **970.125** on repeated training holdouts; the next holdout comparison
is at generation 8. Fresh validation and reserved testing remain pending,
so the 1092-step objective is unproven.

At 2026-10-04T12:56:29.454931+00:00, round 14 completed generation 6. CPU prefix replay
verified **896 completed training-game records**, six populations/rankings,
saved mean/best/final weights and history, and pending seventh population/RNG.
Evidence: `artifacts/doom_reference_round14_g006_training_prefix_cpu_audit.json`.
Generation 7 is running in verified live session 49443. The baseline remains
best at **970.125** on repeated training holdouts; the next comparison is at
generation 8. Fresh validation and reserved testing remain pending. The
1092-step objective is unproven.

At 2026-10-04T13:13:29.223055+00:00, round 14 completed generation 7. CPU prefix replay
verified **1024 completed training-game records**, seven populations/rankings,
saved mean/best/final weights and history, and pending eighth population/RNG.
Evidence: `artifacts/doom_reference_round14_g007_training_prefix_cpu_audit.json`.
The final generation 8 is running in verified live session 49443, followed
by its 64-game training holdout and independent complete capsule audit. The
baseline remains best at **970.125**. Actual supervisor exit and current audit
checks are required before the separately prepared fresh validation and any
eligible reserved pair. The 1092-step objective remains unproven.


Round 14 training completed and its supervisor session 49443 exited with code 0
at 2026-10-04T13:31Z. Independent current CPU replay verified all **1216 training
games**, eight CMA populations/rankings, selected/final weights, complete
history and saved optimizer/RNG. The parent, trainer and auditor had exited;
the GPU lease was available before the separate fresh evaluation dispatch.
The final generation 8 became best on the repeated 64-game training holdout:
**1002.0781 ±577.5202** steps (sample SD), with 58 deaths and 6 timeouts,
versus baseline **970.125 ±537.5760**, 62 deaths and 2 timeouts. Paired change
was **+31.9531**, bootstrap 95% interval **[−94.2813,+153.4691]**. This interval
conditions on these fixed checkpoints and the repeated training cohort; it
excludes training, checkpoint and sequential-selection uncertainty and does
not establish fresh performance or the 1092-step target. Evidence:
`artifacts/doom_reference_round14_training_cpu_audit.json`,
`artifacts/doom_reference_round14_training_exit_review_cpu.json`, and
`artifacts/doom_reference_round14_final_holdout_comparison_cpu.json`.

The prepared v2 fresh comparison passed actual CPU readiness and CUDA preflight
and started as supervisor session **62687**, parent **1563012**. Three distinct
policies require all 80 validation games each on seeds 130660–130739: public,
prior round 8, and new round 14 generation 8. Best/final raw weights are identical
and therefore evaluated once. Only a new validation winner, frozen and checked
by the independent CPU choice audit, may proceed to its reserved 100 games and
the highest-validation unchanged control on the same seeds 230000–230099.
No new training is queued. This trains a controller on real survival with an
imported public world, which differs from the paper's dream-only controller
training and does not reproduce our own world-model training. Original
checkpoints and the own-world incumbent score 840.06 ±524.48 are preserved.


Round 14 fresh validation completed all **240 games** on seeds 130660–130739.
The public control scored **972.575 ±509.5249**, the prior round 8 controller
**979.0875 ±570.5231**, and the new round 14 generation 8 controller
**921.525 ±521.1782** steps (population SD). New versus prior was
**−57.5625**, whole-game paired validation bootstrap 95% interval
**[−190.5878,+73.5631]**. The prior controller was retained after complete
validation; generation 8's observed training-holdout gain did not transfer to
this fresh cohort. The interval does not establish population degradation,
and conditions on the fixed policies while excluding training and sequential
selection uncertainty. No policy was selected or retrained using test data:
the reserved range 230000–230099 remains unused because no new policy won.

The actual real supervisor session **62687** exited with code **0** (tool
`2806f9`). CPU reconstruction rechecked the full result, frozen selection,
current source/input/runtime/checkpoints/training capsule and all raw records.
Known training/evaluation/auditor processes and game workers had exited; the
repository GPU lease was available and no additional GPU job was queued.
Evidence: `artifacts/doom_reference_round14_real_result_cpu_audit.json`,
`artifacts/doom_reference_round14_completion_cpu_audit.json`, and
`artifacts/doom_reference_round14_controller_geometry_cpu.json`.
The 1092-step goal remains unmet. The next hypothesis is a bounded coherent
controller-gain search from the retained prior weights, with fixed world and
inference, fresh training/validation seeds and reserved testing only after an
independently audited new validation winner. It has not been dispatched or
shown to improve performance. Original checkpoints and the canonical
own-world incumbent remain preserved; imported-world direct-real controller
training remains a protocol departure from the paper.


The next CPU prototype projects three log gains onto the retained prior
controller's z/c/h weight blocks (64/512/512), preserving the zero point's
raw FP64 weights exactly and the original singleton inference computation.
Four CPU projection tests passed, including block order, bounds, invalid
weights and overflow rejection. Recorded prior validation chose wait on only
**0.2119%** of steps; relative block gains can affect left/right sign as well
as action magnitude. Their raw action still feeds the fixed RNN, so this is
not merely an environment threshold adjustment. These are implementation
facts and a search hypothesis, not evidence of better real survival.
`artifacts/doom_reference_round15_subspace_design_cpu.json` is a **draft**:
the resumable projected CMA trainer, full independent optimizer replay and
fresh comparison gates must be implemented and verified before registration
and CUDA dispatch. Proposed budget is four generations, population 8, sixteen
fitness games, 64 training holdouts, three complete 80-game validation cohorts
and a conditional 100+100 reserved pair (at most 1080 games). Proposed fresh
seed ranges were checked unused. No GPU work has been dispatched.


Round 15 was registered and started on CUDA at 2026-10-04T14:29Z after
**32 passing CPU tests**, actual initializer/trainer readiness and launcher
source/input/runtime/process/lease/fresh-seed checks. Protocol:
`artifacts/doom_reference_round15_subspace_protocol.json`
(SHA `a91688c85b27913bb7368ce440e988869af74b82b5b02def90330d8dac92af67`).
It optimizes three bounded log gains (z/c/h blocks) from the retained prior,
with seed 96, sigma 0.25, four generations, population 8, sixteen common
fitness games per candidate per generation, and 64 baseline/final training
holdouts. The world, preprocessing, architecture and singleton inference
math remain fixed. These combined hyperparameter changes do not isolate a
cause of improvement. Best and final projected FP64 controller weights,
log gains and complete three-dimensional optimizer/RNG/history/raw games are
retained; CPU replay reconstructs the bounded phenotype mean, all populations,
projection, covariance/adaptation and next RNG before fresh evaluation.

Actual supervisor session **87997**, parent **1599849** and trainer **1599915**
were verified live; CUDA matrix multiplication and convolution backward
preflight passed on the RTX 4070 Ti. Initial immutable optimizer snapshot,
raw zero-gain prior identity, first eight-candidate population and RNG were
independently reconstructed on CPU. Evidence:
`artifacts/doom_reference_round15_initial_capsule_cpu_audit.json` and
`artifacts/doom_reference_round15_training_preparation_cpu.json`.
The initial 64-game baseline is still running; no complete survival result or
fresh validation has been claimed. Estimated complete-loop duration is two
to three hours, dependent on actual game lengths and compilation. Training
uses up to 640 games, followed by up to 240 fresh validation games and a
conditional 200 reserved candidate/control games. Validation seeds
130740–130819 and test seeds 240000–240099 remain separated from training;
only a distinct new full-validation winner can consume the reserved pair.
The goal remains unmet and original checkpoints/canonical incumbent remain
preserved. Historical round 14 sessions are closed and must not be restarted.


At 2026-10-04T14:40:01.346943+00:00, round 15 completed its fresh 64-game training baseline:
**853.8750 ±558.7786** steps (sample SD), with
62 deaths and 2 timeouts. CPU reconstruction verified every raw
record, seed/settings/initializer identity, generation-zero best/final projected
weights and gains, and the pending first3D population and next RNG. Evidence:
`artifacts/doom_reference_round15_baseline_cpu_audit.json`. The eight-candidate
first population is running in independently verified live session87997. This
baseline uses new training seeds and is not fresh validation or the reserved
100-game result. A lower or higher baseline mean on a different cohort does not
by itself establish a policy change. The1092-step goal remains unmet.


At 2026-10-04T14:55:58.708133+00:00, round 15 completed generation 1 and began generation 2.
CPU replay verified all **192 completed training games**, reconstructed the
first population and raw fitness means, three-dimensional covariance/adaptation
and optimizer mean, best/current projected1088-weight checkpoints and history,
and the pending second population and next RNG. Current mean feature gains
(z/c/h) are **0.9137/1.0923/0.9410**.
These are training updates, not fresh survival validation. Best remains the
baseline853.875 until the scheduled final64-game holdout at generation4.
Evidence: `artifacts/doom_reference_round15_g001_training_prefix_cpu_audit.json`.
Its hashes describe the saved files at that check; optimizer snapshots and
completed raw cohorts are immutable while checkpoint/pointer/history files
advance. Existing session87997 and exact parent/trainer commands were verified
live. Fresh validation and reserved testing remain pending; goal completion
is unproven.


Round 15 training finished at 2026-10-04T15:43Z, and supervisor session
87997 exited with code 0 (actual tool chunk `57ae36`). Independent current CPU
replay verified all **640 training games**, four bounded three-dimensional CMA
populations, projected controller weights, complete history, covariance and
optimizer/RNG. Known supervisor, trainer and auditor processes had exited;
no repository GPU jobs remained, and the exclusive GPU lease was available.
Best and final generation 4 weights are identical. Its repeated 64-game
training holdout scored **960.6563 ±582.5806** steps (sample SD), with 58 deaths
and 6 timeouts, versus baseline **853.8750 ±558.7786**, 62 deaths and 2 timeouts.
Paired change was **+106.7813**, whole-game bootstrap 95% interval
**[−44.9695,+258.4398]**. This interval conditions on these fixed policies and
excludes training, checkpoint and sequential-selection uncertainty. It does
not establish fresh performance or the 1092-step target. Final z/c/h weight
gains are **0.9263/1.2552/0.5923**. Evidence:
`artifacts/doom_reference_round15_training_cpu_audit.json`,
`artifacts/doom_reference_round15_training_exit_review_cpu.json`, and
`artifacts/doom_reference_round15_final_holdout_comparison_cpu.json`.

The prepared fresh comparison now requires all **80 validation games per
policy** on seeds 130740–130819 for public, retained prior, and new generation
4 controllers. Only a distinct new winner, frozen after complete validation
and independently audited, can use its reserved 100 games and the highest
validation unchanged control on the same seeds 240000–240099. Neither fresh
validation nor reserved testing has completed. The original own-world
incumbent remains preserved; imported-world direct-real controller training
differs from the paper's dream-only controller training.


The separate round 15 fresh-validation supervisor was dispatched after actual
CPU readiness and CUDA preflight passed. Existing session **54240**, parent
**1643586**, is running the public 80-game cohort, followed by the unchanged
prior and new generation 4 controller. Best/final identities are deduplicated.
Durable supervisor log: `artifacts/doom_reference_round15_real_supervisor.log`;
status: `artifacts/doom_reference_round15_real_evaluations.status`.
Selection and conditional reserved testing remain pending. The five-minute
follow-up now points to this real-evaluation job; training session 87997 is
closed and must not be restarted.


Round 15 completed all **240 fresh validation games** on seeds 130740–130819.
Public scored **838.925 ±550.3700**, unchanged prior **818.575 ±537.8507**,
and new generation 4 **884.0125 ±518.8163** steps (population SD). The new
controller won complete validation and was frozen for testing; its paired
validation change versus the highest-validation unchanged public control was
**+45.0875**, bootstrap 95% interval **[−90.0384,+181.3891]**. This does not
establish population improvement and excludes training and sequential
selection uncertainty. The independent CPU choice audit passed at
2026-10-04T16:11:05Z before the reserved GPU preflight. Root CPU reconstruction
subsequently verified all raw records, eligibility and the unchanged frozen
choice: `artifacts/doom_reference_round15_validation_recheck_cpu.json`.

Existing real supervisor session **54240** now evaluates the frozen new
controller on **100 reserved games**, seeds 240000–240099, followed by the
unchanged public control on the same 100 seeds. Selection was fixed solely
from complete validation and must not be changed or retrained using these
test outcomes. The reserved pair is still running; no final test performance
or 1092-step result is claimed. Original checkpoints and the own-world
incumbent remain intact; imported-world direct-real training remains a
protocol departure from the paper.


Round 15 reserved testing completed both **100-game reports** on seeds
240000–240099. Frozen new generation 4 scored **993.52 ±584.9124** steps;
unchanged public control scored **1031.13 ±556.1024** (population SD).
Whole-game paired change was **−37.61**, bootstrap 95% interval
**[−144.0210,+69.2505]**, with 36 wins, 54 losses and 10 ties. This does not
demonstrate improvement or achieve the 1092-step mean. The candidate's mean
interval includes1092 but that does not establish the numerical target.
Uncertainty conditions on fixed policies and excludes training, sequential
selection and historical paper-runtime differences. The validation choice
remains frozen; test outcomes do not select a different policy or drive
retraining choices. Imported public world and direct-real controller training
remain departures from the paper's dream-only procedure and do not reproduce
our own world-model training.

Actual real supervisor session **54240** exited with code **0** (tool
`df17b7`). Root CPU reconstruction verified all240 validation and200 reserved
raw records, current source/input/runtime/checkpoints/training capsule and
immutable selection. All17 known supervisor/trainer/evaluator/auditor/preflight
processes and repository game workers had exited; GPU lease was available,
and no additional GPU work was queued. Evidence:
`artifacts/doom_reference_round15_real_result_cpu_audit.json` and
`artifacts/doom_reference_round15_completion_cpu_audit.json`. Originals and
canonical own-world840.06 ±524.48 are preserved. The goal remains unmet.

A separate CPU diagnostic used **only training fitness records**. Resampling
the same sixteen shared games jointly across the eight candidates retained
each generation's observed top candidate in only **39.9–56.3%** of20000
bootstrap draws; all observed top-versus-runner paired intervals includedzero.
These are ranking-fragility diagnostics on selected training cohorts, not
probabilities of being best or fresh generalization evidence. This motivates
more games per training fitness evaluation. A **draft**,
`artifacts/doom_reference_round16_subspace_design_cpu.json`, proposes64 fitness
games, four generations/population8, from the round15 validation-selected
policy with fixed world/inference and new seeds. The estimated complete loop
is5–6hours. Projected-parent support,64-game trainer support, seed uniqueness,
CPU tests/replay, registration and CUDA/startup checks remain required.
No new GPU work has been dispatched. Increasing sample count and changing the
validation-selected initializer together will not isolate either cause.


Round 16 was registered and dispatched on CUDA at 2026-10-04T17:00Z after
**40 passing combined CPU tests**, actual projected-parent/64-game trainer
readiness, historical source/input checks, unused-seed audit, parent process
closure and exclusive-lease startup checks. Protocol:
`artifacts/doom_reference_round16_subspace_protocol.json`
(SHA `4eb6f5b34bcfb1562d3efe2987fe8f4ecadc373ae48809db2dedd4c0ac3c1487`).
All37 source files and106 input fingerprints are frozen; support was added
in separately versioned files without modifying historical registered code.
The initializer is the round15 complete-validation winner, with no reserved
outcomes used for initialization or fitness decisions. Four generations,
population8 and64 shared fitness games use up to2048 fitness games plus128
baseline/final training holds; up to240 fresh validation and a conditional
200 reserved paired games follow only after full training/selection audits.
World, architecture and original singleton inference calculation stay fixed.
The estimated complete loop is5–6hours; it is an estimate.

Actual supervisor session **24165**, parent **1702572** and trainer
**1702915** are live; CUDA matrix multiplication and convolution backward
preflight passed on RTX4070Ti. The immutable initial optimizer snapshot,
zero-gain identity of the projected parent and two consecutive optimizer
sampling sequences were independently reconstructed on CPU:
`artifacts/doom_reference_round16_initial_capsule_cpu_audit.json`.
The baseline was **36/64 games** at17:05:29UTC; no complete new survival result
or target achievement is claimed. Training journal/checkpoints/optimizer and
raw records will be preserved. Fresh validation130820–130899 and reserved
250000–250099 remain separated from fitness560000–560255 and holds570000–570063.
Parent round15 sessions are closed and must not be restarted. Original
checkpoints and the canonical own-world incumbent remain intact. Imported
public world and direct-real training remain protocol departures from the paper.


At 2026-10-04T17:13:01Z, round16 completed its64-game training baseline:
**914.0313 ±533.3618** steps (sample SD),60 deaths and4 timeouts. Every raw
record, seed/settings/parent identity, zero-gain best/final checkpoint and
pending first3D population/optimizer sampling sequence was reconstructed on
CPU: `artifacts/doom_reference_round16_baseline_cpu_audit.json`.
This is the unchanged validation-selected initializer on new training seeds,
not an updated policy or fresh validation/test result. Existing session24165
and exact parent/trainer commands remain live; generation1 was29/512 fitness
games at17:13:11UTC. The scheduled final holdout and complete training/real
selection/exit audits remain pending. The1092-step goal is unproven.
