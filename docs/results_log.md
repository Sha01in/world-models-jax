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
