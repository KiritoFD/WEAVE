# ICME 2027 Narrative & Rebuttal Blueprint (`STORY.md`)

## 1. Title & Core Narrative

- **Title**:
  > **PRISM: Breaking Source-Style Lock to Redefine the Content–Style–Speed Frontier**

- **Three Attack Propositions (Introduction)**:
  1. **Scaling Does Not Solve Style Transfer ($260\times$ Self-Paired Scaling Paradox, Fig. 2a)**:
     Scaling a self-paired latent flow U-Net by $260\times$ ($0.13\text{M}\to 34.2\text{M}$) improves unconditional generation ($\tau_0=0$) from $20.0\%$ to $49.3\%$, yet monotonically *degrades* cross-style SDEdit ($\tau_0=0.7$) from $10.67\%$ to $6.67\%$ ($59\%$ predicted back as source style $y_s$). Even $\tau_0=0$ flows conditioned on $64\times 64$ edges stall at $18.67\%$ because self-paired training ($x_s\mapsto x_s$) inverts 2D stroke cadence back into $y_s$.
  2. **Reconstruction Fidelity Can Fight Stylization (Spectral-Bottleneck Decoder Paradox, Fig. 2b)**:
     More capacity can preserve the wrong information not only in the flow, but also in the decoder. Full $49.5\text{M}$ SD-VAE synthesis filters memorize source-domain texture priors; structured $78.5\%$ channel pruning (**PRISM-Dec**, $10.66\text{M}$) accelerates decoding by $3.15\times$ on RTX 3060 (`214.4 -> 68.1 ms/img` at `B8`) while *increasing* cross-style Top-1 by $+3.17\text{pp}$ ($p<0.01$).
  3. **Evaluation Mirages Obscure True Specificity (Fig. 2c)**:
     Standard diagonal whitening retains $92.33\%$ off-diagonal 2nd-order Gram style leakage, while post-hoc subband scalers score higher under counterfactual *wrong* target labels than true ones ($13.0\%$ vs. $12.3\%$).

- **Three Headline Contributions**:
  1. *A Scaling Paradox in Latent Style Transfer*: Theoretical (Prop. 1) and large-scale empirical identification of **Source-Style Lock** across self-paired flows, feature normalizers, and reconstruction decoders.
  2. *PRISM: Structure-Invariant, Style-Directed Latent Transport*: Orthogonal **Quotient–Fiber Duality** $(\Pi_\perp,\Pi_\parallel)$ (Stiefel polar projection $Q_{32}$ clamping 2nd-order Gram leakage from $92.33\%$ to $20.00\%$ exact uniform-chance floor + complementary fiber $\Pi_\parallel$ with $97.33\%$ style accuracy forming a source-label-free target chord) paired with controlled transport $\mathcal{T}$ and spectral-bottleneck decoding $\mathcal{D}_\star$.
  3. *A New Content–Style–Compute Frontier (Unified RTX 3060 12GB)*: Spanning real-time 1-step (**PRISM-1**: $58.5\,\text{ms}$ cached / $167.5\,\text{ms}$ max-batch RGB$\to$RGB = $5.97\,\text{img/s}$ E2E on 12GB 3060, $44.33\%$ Top-1 beating Seedream 4.5 and StyTr$^2$ on all 5 metrics), high-content half-NFE hybrid (**PRISM-HC**: `B4 + cs_j + D6H + PRISM-1`, $0.7723$ DINO-C, $0.4564$ DINO-S, $0.4783$ LPIPS, $297.5\,\text{ms}$ E2E = $3.36\,\text{img/s}$), and generative brushstroke flow (**PRISM-Flow**: $60.50\%$ Top-1, $0.5672$ DINO-S, $0.6584$ Affinity, $680.3\,\text{ms}$ E2E = $1.47\,\text{img/s}$ or $296.3\,\text{ms}$ = $3.38\,\text{img/s}$ with 10-step `cs_j`, surpassing diffusion cross-attention StyleGallery at $30.4\,\text{s/img}$ under $<3.9\,\text{GB}$ VRAM).

---

## 2. Figure Design Summary

- **Figure 1 (`fig1_prism_pareto.pdf`) — Computational Paradigm Confrontation**:
  - Three explicit paradigms: `Diffusion-Heavy` ($5.0\text{--}30.4\,\text{s/img}$), `Conventional Lightweight` ($0.2\text{--}3.0\,\text{s/img}$), and `PRISM` (all measured on RTX 3060).
  - Bubble radius mapped to measured RTX 3060 max-batch E2E throughput ($\text{img/s}$), with explicit latency callouts ($\text{s/img}$) on slow baselines (`StyleGallery 30.4 s/img`, `StyleShot 24.5 s/img`, `StyleID 5.0 s/img`, `StyTr2 3.0 s/img`).
  - Simplified single orange-red global Pareto frontier per subplot highlighting 6 representative operating points without internal `V22/V23/V25` codenames.
- **Figure 2 (`fig2_three_paradoxes.pdf`) — More Capacity Can Preserve the Wrong Information**:
  - Unifies the $260\times$ Flow Scaling Paradox, the $78.5\%$ Spectral-Bottleneck Decoder Paradox ($3.15\times$ on RTX 3060), and the Counterfactual Specificity Audit.
- **Figure 3 (`fig3_prism_architecture.pdf`) — Blocking Source-Style Backflow**:
  - Top banner and stage badges highlight the causal mechanism: `Source Texture Leakage (92.3% -> 20.0%) -> Invariant Layout \Pi_\perp -> Target Style Chord e(x_s -> \tau) -> Controlled Transport \mathcal{T} -> Spectral-Bottleneck \mathcal{D}_\star`.
- **Figure 4 (`fig5_qualitative.pdf`) — Six-Column Qualitative Confrontation**:
  - Six spacious columns (`Source`, `StyleShot`, `StyleGallery`, `Seedream 4.5`, `PRISM-HC`, `PRISM-Flow`) proving `PRISM-HC` content fidelity and `PRISM-Flow` brushstroke strength directly against top external baselines.

---

## 3. Rebuttal-Proof Defense Matrix

1. **Protocol-Cached vs. Full RGB$\to$RGB End-to-End Latency (All RTX 3060 12GB)**:
   - Table I and Section V report *both* protocol-cached transport+decode ($58.5\,\text{ms}$ for PRISM-1, $189.4\,\text{ms}$ for PRISM-HC, $541.0\,\text{ms}$ for PRISM-Flow) and full $512^2$ RGB$\to$RGB end-to-end latency including A0 VAE encode, DINOv2 extraction, anchor, transport, and S10 decode on a single consumer 12GB RTX 3060 ($167.5\,\text{ms}$ / $5.97\,\text{img/s}$ for PRISM-1, $297.5\,\text{ms}$ / $3.36\,\text{img/s}$ for PRISM-HC, $680.3\,\text{ms}$ / $1.47\,\text{img/s}$ for PRISM-Flow, all under $<3.9\,\text{GB}$ VRAM).
2. **Three-Factor Attribution of `PRISM-HC` (`B4 + cs_j + D6H`)**:
   - Under unified random seed (`20261185`) and 150-source paired cluster bootstrap CIs at $\gamma=0.55$, we isolate Anchor (`B4 vs T4`: $\Delta\text{DINO-C}=+0.0362$, $\Delta\text{LPIPS}=-0.0255$, contributing $\sim 80\%$ of the high-content gain), Solver (`D6H vs D6`: $+0.0099$ DINO-C, $-0.0060$ LPIPS), and CFG (`cs_j vs full`: $+0.0018$ DINO-S while halving flow NFE and giving $2.05\times$ flow speedup on RTX 3060).
3. **When is 4-Branch `full` CFG Still Needed?**:
   - Honestly reported in Section IV-C and V: while 2-branch `cs_j` matches or exceeds 4-branch `full` globally at $2.33\times$ faster batched flow speed, `full`'s unconditional style prior term $w_p(v_{s0}-v_{00})$ is specifically beneficial for de-cluttering dense source scenes into sparse targets (*Impressionism$\to$Minimalism*: $86.7\%$ vs. $30.0\%$ at D5).
