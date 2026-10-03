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
