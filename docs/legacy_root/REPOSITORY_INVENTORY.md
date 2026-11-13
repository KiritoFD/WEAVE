> **已归档（2026-11-14 仓库重组）**：论文快照期/阶段性文档，现状见 [docs/README.md](../README.md)。

# WEAVE Repository Inventory

**Generated:** 2026-07-25 08:32:43
**Location:** I:\Github\Latent_Style\WEAVE\

## 1. Overview

This repository contains the complete WEAVE (Wavelet-based Entanglement-Augmented Vision Exchange)
style transfer project: code, datasets, experimental results, checkpoints, and paper.

## 2. Top-level Directory Structure

| Directory | Description |
|-----------|-------------|
| aaai2027_v4/ | AAAI 2027 paper (LaTeX source, PDF, figures) |
| archive/ | Archived experiments and historical data (read-only backup) |
| archives/ | Legacy results archives (R5-WikiArt legacy) |
| AuthorKit27/ | AAAI 2027 author kit (sty/bst files) |
| config/ | MCP porter config |
| configs/ | Model configuration files |
| data/ | Processed datasets (train/test/latents/onnx) |
| datasets/ | Raw dataset copies |
| docs/ | Project documentation (exp/baseline/math/tools) |
| eval_outputs/ | Evaluation output files |
| exp/ | Active experiments (paper-cited) |
| experiments/ | Legacy experiment configs |
| logs/ | Training and evaluation logs |
| maths/ | Mathematical analysis documents |
| probes/ | Model probe scripts |
| results/ | Evaluation results, metrics, tables |
| runs/ | Training run outputs (CSV checkpoints) |
| scripts/ | Utility and build scripts |
| state/ | DINO metrics, baseline data, progress tracking |
| tests/ | Test scripts |
| tools/ | Data preparation, inference, benchmark tools |
| utils/ | Utility functions (dataset, metrics, training) |
| visualization/ | Visualization assets |

## 3. Detailed Directory Contents

### aaai2027_v4/

- [FILE] .$frame.drawio.bkp (1,016.9 KB)
- [FILE] _tmp_recompute_svg.py (1.6 KB)
- [FILE] aaai_arch_diagram_v16_staggered_bundle.300dpi.png (677.7 KB)
- [FILE] aaai_arch_diagram_v16_staggered_bundle.300dpi_edited.png (677.7 KB)
- [FILE] aaai_arch_diagram_v16_staggered_bundle.drawio.png (677.7 KB)
- [FILE] aaai2027.bst (29.5 KB)
- [FILE] aaai2027.sty (16.5 KB)
- [FILE] b1.log (7.6 KB)
- [FILE] b2.log (6.1 KB)
- [FILE] b3.log (8.4 KB)
- [FILE] b4.log (7.5 KB)
- [FILE] bib.log (146 B)
- [FILE] bibtex_out.txt (146 B)
- [FILE] build.ps1 (871 B)
- [FILE] build_b.log (146 B)
- [FILE] build_bib.txt (146 B)
- [FILE] build_log1.txt (10.8 KB)
- [FILE] build_log2.txt (10.8 KB)
- [FILE] build_log3.txt (12.3 KB)
- [FILE] build_now.txt (12.3 KB)
- [FILE] build_p1.log (5.0 KB)
- [FILE] build_p2.log (6.3 KB)
- [FILE] build_p3.log (6.3 KB)
- [FILE] build_supplement_pass1.log (16.2 KB)
- [FILE] build_supplement_pass2.log (16.2 KB)
- [FILE] compile_layout.log (8.2 KB)
- [FILE] compile_method.log (9.7 KB)
- [FILE] compile_restore.log (8.3 KB)
- [FILE] compile_rw.log (360 B)
- [DIR ] fig_data (458.4 KB)
- [FILE] fig_distinct5_page1_summary.pdf (105.2 KB)
- [FILE] fig_distinct5_page1_summary.png (248.9 KB)
- [FILE] fig_distinct5_page1_summary_no_tgt.pdf (82.1 KB)
- [FILE] fig_distinct5_page1_summary_no_tgt.png (257.5 KB)
- [FILE] fig_fiber_path_diagnostics.png (90.1 KB)
- [FILE] fig_method_probes.pdf (36.3 KB)
- [FILE] fig_method_probes.png (102.4 KB)
- [FILE] fig_page1_a.png (1.1 MB)
- [FILE] fig_page1_a.svg (153.2 KB)
- [FILE] fig_page1_b.png (1.8 MB)
- [FILE] fig_page1_b.svg (33.8 KB)
- [FILE] fig_page1_stacked.png (256.4 KB)
- [FILE] fig_pixel_vs_latent_affine.pdf (13.9 KB)
- [FILE] fig_pixel_vs_latent_affine.png (52.7 KB)
- [FILE] fig_pixel_vs_latent_no_frequency.pdf (20.0 KB)
- [FILE] fig_pixel_vs_latent_no_frequency.png (184.7 KB)
- [FILE] fig_pixel_vs_latent_one_column.pdf (17.2 KB)
- [FILE] fig_pixel_vs_latent_one_column.png (129.6 KB)
- [FILE] fig_probe_combined.pdf (12.3 KB)
- [FILE] fig_probe_combined.png (29.9 KB)
- [FILE] fig_probe_frequency.pdf (9.9 KB)
- [FILE] fig_probe_frequency.png (26.2 KB)
- [FILE] fig_probe_style.pdf (9.5 KB)
- [FILE] fig_probe_style.png (26.4 KB)
- [FILE] fig_probe_swd_separation.pdf (11.3 KB)
- [FILE] fig_probe_swd_separation.png (20.3 KB)
- [FILE] fig_signal_level.pdf (17.6 KB)
- [FILE] fig_signal_level.png (88.0 KB)
- [FILE] fig_teaser_comparison.300dpi.png (1.0 MB)
- [FILE] fig_teaser_comparison.pdf (144 B)
- [FILE] fig_teaser_comparison.png (797.8 KB)
- [FILE] frame.drawio (1,017.0 KB)
- [FILE] frame.png (1.9 MB)
- [FILE] frame_original_preview.png (441.4 KB)
- [FILE] frame723.png (3.2 MB)
- [FILE] framework_sfm_main.pdf (44.6 KB)
- [FILE] framework_sfm_main.png (410.8 KB)
- [FILE] gen_figures.ps1 (1.2 KB)
- [FILE] k_sensitivity_summary.json (1.5 KB)
- [FILE] make_radar.py (22.1 KB)
- [FILE] make_radar_metric_blocks.py (18.8 KB)
- [FILE] make_supplement_figures.py (12.6 KB)
- [FILE] pageimg-2.png (218.9 KB)
- [FILE] pageimg-3.png (130.0 KB)
- [FILE] pageimg-4.png (219.7 KB)
- [FILE] paper - 副本.pdf (9.6 MB)
- [FILE] paper.aux (8.1 KB)
- [FILE] paper.bbl (16.1 KB)
- [FILE] paper.blg (916 B)
- [FILE] paper.fdb_latexmk (12.1 KB)
- [FILE] paper.fls (12.0 KB)
- [FILE] paper.log (19.0 KB)
- [FILE] paper.pdf (9.6 MB)
- [FILE] paper.synctex.gz (157.5 KB)
- [FILE] paper.tex (38.9 KB)
- [FILE] pass3.log (4.9 KB)
- [FILE] pdflatex_pass1.log (5.5 KB)
- [FILE] pdflatex_pass2.log (5.5 KB)
- [FILE] pdflatex30736.fls (212 B)
- [FILE] pdflatex41876.fls (241 B)
- [FILE] pdflatex65928.fls (212 B)
- [FILE] pdflatex65992.fls (212 B)
- [FILE] pdflatex75312.fls (200 B)
- [FILE] pdflatex75760.fls (241 B)
- [FILE] pdflatex84600.fls (200 B)
- [FILE] plot_method_probes.py (6.4 KB)
- [FILE] plot_page1_a.py (20.6 KB)
- [FILE] plot_page1_b.py (11.9 KB)
- [FILE] plot_page1_summary.py (27.2 KB)
- [FILE] plot_signal_level.py (4.6 KB)
- [FILE] radar_baselines_14.png (771.1 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_fixedrange.png (760.1 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_gamma075.png (684.3 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_minmax.png (683.6 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_robustbreak.pdf (84.1 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_robustbreak.png (341.5 KB)
- [FILE] radar_metric_blocks_A_clip_dinos_softminmax.png (687.3 KB)
- [FILE] radar_metric_blocks_A_dinos_raw01.png (658.8 KB)
- [FILE] radar_metric_blocks_A_dinos_sqrt.png (685.0 KB)
- [FILE] README.md (2.3 KB)
- [FILE] refs.bib (21.3 KB)
- [FILE] ReproducibilityChecklist.aux (32 B)
- [FILE] ReproducibilityChecklist.fdb_latexmk (6.6 KB)
- [FILE] ReproducibilityChecklist.fls (6.4 KB)
- [FILE] ReproducibilityChecklist.log (10.9 KB)
- [FILE] ReproducibilityChecklist.pdf (94.2 KB)
- [FILE] ReproducibilityChecklist.tex (7.9 KB)
- [FILE] ReproducibilityChecklist_submission.pdf (93.5 KB)
- [FILE] run_method_probes.py (10.8 KB)
- [FILE] supplement.aux (1.7 KB)
- [FILE] supplement.bbl (0 B)
- [FILE] supplement.blg (1018 B)
- [FILE] supplement.fdb_latexmk (14.7 KB)
- [FILE] supplement.fls (17.6 KB)
- [FILE] supplement.log (38.3 KB)
- [FILE] supplement.pdf (1,013.2 KB)
- [FILE] supplement.tex (68.6 KB)
- [DIR ] supplement_figures (2.1 MB)
- [FILE] supplement_submission.pdf (766.0 KB)
- [FILE] SUPPLEMENTARY_MATERIAL.md (11.0 KB)
- [FILE] teaser.png (5.1 MB)
- [FILE] teasr.drawio (624.7 KB)
- [DIR ] tmp (23.5 MB)
- [DIR ] tmp_extract (165.9 KB)
- [FILE] update_paper_figure3.py (6.5 KB)
- [FILE] weave_architecture.drawio (31.1 KB)
- [FILE] weave_architecture_preview.png (446.3 KB)
- [FILE] weave_architecture_preview2.png (347.3 KB)

### archive/

- [DIR ] archives (5.87 GB)
- [DIR ] AuthorKit27 (350.7 KB)
- [DIR ] dinov2_weights (196 B)
- [DIR ] eval_cache (5.87 GB)
- [DIR ] exp (12.56 GB)
- [DIR ] exp_sandbox (41.3 MB)
- [DIR ] experiments (1.15 GB)
- [DIR ] logs (6.2 MB)
- [DIR ] maths (60.5 KB)
- [DIR ] probes (27.4 KB)
- [DIR ] runs_ablation_endpoint_only (86.1 MB)
- [DIR ] runs_ablation_stepwise (81.1 MB)
- [DIR ] runs_analysis (2.8 MB)
- [DIR ] runs_arch_hf_oriented_nohh_15ep (282.1 KB)
- [DIR ] runs_cache (3.46 GB)
- [DIR ] runs_probe_whitening (1.4 MB)
- [DIR ] runs_probe_whitening_fixed (4.3 MB)
- [DIR ] runs_probe_whitening_nopreserve (2.9 MB)
- [DIR ] runs_probe_whitening_nopreserve3 (4.3 MB)
- [DIR ] runs_probe_whitening_preserve (2.9 MB)
- [DIR ] runs_probe_whitening_preserve3 (4.3 MB)
- [DIR ] runs_submission (494.5 MB)
- [DIR ] stabilityai (60 B)
- [DIR ] subpack (9.25 GB)
- [DIR ] tests (44.9 KB)
- [DIR ] tools (161.3 KB)
- [DIR ] top_level (29.51 GB)
- [DIR ] visualization (50.3 KB)

### archives/

- [DIR ] results_R5-WikiArt_legacy_20260721 (5.55 GB)

### data/

- [DIR ] onnx_exported (319.5 MB)
- [FILE] README.md (1.0 KB)
- [DIR ] taesd (9.3 MB)
- [DIR ] test (41.5 MB)
- [DIR ] train (638.1 MB)
- [DIR ] train_d5 (5.39 GB)
- [DIR ] train_sdxl (313.3 MB)
- [DIR ] train_taesd (313.3 MB)

### datasets/

- [DIR ] 256 (342.1 MB)
- [DIR ] clip-feats-vitb32 (1.27 GB)
- [DIR ] d5 (1.77 GB)
- [DIR ] distinct5_512 (3.75 GB)
- [DIR ] distinct5_512_latents_ema (332.5 MB)
- [DIR ] distinct5_512_sdxl (300.7 MB)
- [DIR ] distinct5_512_train_zstar_src (28.3 MB)
- [DIR ] distinct5_512_vavae (634.5 MB)
- [DIR ] eval (116.4 MB)
- [DIR ] fewshot6_512 (42.5 MB)
- [DIR ] fewshot6_512_latents_ema (314.3 MB)
- [DIR ] fewshot8_512_latents_ema (660.9 MB)
- [DIR ] latent_cyclegan (850.4 KB)
- [DIR ] latent-256 (177.7 MB)
- [DIR ] legacy256_overfit50 (342.1 MB)
- [DIR ] legacy256_overfit50_vavae (664.1 MB)
- [DIR ] other5_512 (108.2 MB)
- [DIR ] other5_reg_latents_ema (4.7 MB)
- [DIR ] r5 (1.37 GB)
- [FILE] r5_train_raw.zip (1.31 GB)
- [FILE] README.md (2.6 KB)
- [DIR ] style_data (385.6 MB)
- [DIR ] wikiart_distinct5_samam_512_latents_ema (315.0 MB)
- [DIR ] wikiart_random20_512 (3.38 GB)
- [DIR ] wikiart_random20_512_latents_ema (1.30 GB)
- [DIR ] wikiart_stress_splits_512 (4.62 GB)
- [DIR ] wikiart512_5style (5.47 GB)

### docs/

- [DIR ] 713 (121.5 KB)
- [DIR ] 79 (31.8 KB)
- [FILE] architecture_diagram.svg (7.9 KB)
- [DIR ] archive (156.4 MB)
- [DIR ] baseline (10.6 KB)
- [DIR ] delivery (12.6 KB)
- [DIR ] dino_s_break (27.0 KB)
- [DIR ] exp (282.6 KB)
- [DIR ] experiments (38.4 KB)
- [DIR ] latent_migration (11.1 KB)
- [DIR ] math (16.6 KB)
- [DIR ] model_probe (12.7 MB)
- [FILE] onnx_export_and_trt.md (1.9 KB)
- [DIR ] plans (7.7 KB)
- [FILE] README.md (2.1 KB)
- [DIR ] reproduction (168.9 KB)
- [DIR ] tools (15.4 KB)
- [FILE] vae_encode_onnx_trt.md (7.7 KB)

### eval_outputs/

- [DIR ] d5_750_baselines (1.2 MB)
- [DIR ] d5_750_endpoint_only (85.1 MB)

### exp/

- [FILE] 710_baseline_identity_dino_canonical_v2.json (232 B)
- [FILE] 710_baseline_identity_dino_canonical_v2.log (1.9 KB)
- [DIR ] 710_baseline_weave_paper_reval (81.6 MB)
- [FILE] 710_baseline_weave_paper_reval_dino.log (973 B)
- [FILE] 710_baseline_weave_paper_reval_dino_canonical.log (654 B)
- [FILE] 710_baseline_weave_paper_reval_dino_canonical_v2.log (1.5 KB)
- [FILE] 710_baseline_weave_paper_reval_dino_fixed.log (832 B)
- [FILE] 710_baseline_weave_paper_reval_eval.log (3.6 KB)
- [FILE] 710_candidate_exp_FCSB_early_clean_base_v2_local_full_eval_epoch_0010_full.csv (170.2 KB)
- [FILE] 710_candidate_exp_FCSB_early_clean_base_v2_local_full_eval_epoch_0010_full.json (652 B)
- [FILE] 710_candidate_exp_FCSB_early_clean_base_v2_relu2_full_eval_epoch_0003_full.csv (170.3 KB)
- [FILE] 710_candidate_exp_FCSB_early_clean_base_v2_relu2_full_eval_epoch_0003_full.json (652 B)
- [FILE] 710_infra_bench_bridge_s1.log (388 B)
- [FILE] 710_infra_bench_bridge_s12.log (390 B)
- [FILE] 710_infra_bench_bridge_s8.log (680 B)
- [FILE] 710_infra_bench_decode_s8_compile.log (1.0 KB)
- [FILE] 710_infra_bench_decode_s8_eager.log (930 B)
- [FILE] 710_infra_bench_decode_s8_eager_localfirst.log (634 B)
- [FILE] 710_infra_bench_decode_s8_torchscript.log (709 B)
- [FILE] 710_infra_bench_end2end_s1_ts_png.log (860 B)
- [FILE] 710_infra_bench_formal_pipeline_8step.log (2.5 KB)
- [FILE] 710_infra_bench_formal_pipeline_8step_png.log (2.5 KB)
- [FILE] 710_infra_bench_formal_pipeline_smoke.log (1.7 KB)
- [FILE] 710_infra_export_vae_ts.log (1.3 KB)
- [FILE] 710_infra_t11_cpu_1ep_console.log (3.7 KB)
- [FILE] 710_infra_t11_cudnn_1ep_console.log (3.7 KB)
- [FILE] 710_infra_t11_distinct5_5ep_console.log (7.0 KB)
- [FILE] 710_infra_t11_power80_console.log (3.9 KB)
- [FILE] 710_infra_t11_power80_gpu.csv (4.0 KB)
- [FILE] 710_infra_t11_preload_1ep_console.log (3.8 KB)
- [FILE] 710_infra_t11_profile12_console.log (17.8 KB)
- [FILE] 710_infra_t11_syncfix_1ep_console.log (3.7 KB)
- [DIR ] a1_reference_only (228.0 KB)
- [DIR ] a3_sweep (411.9 KB)
- [DIR ] ablation_v2 (3.6 KB)
- [DIR ] b2_qualitative (6.2 MB)
- [DIR ] b3_failure_cases (4.8 MB)
- [DIR ] b4_reference_stability (815.5 KB)
- [DIR ] b5_compute_cost (6.7 KB)
- [DIR ] dino_avg_vs_max (5.7 KB)
- [DIR ] dino_s_break (4.00 GB)
- [DIR ] dwt_route_distinct5 (187.5 MB)
- [DIR ] eval_all (22.3 KB)
- [FILE] evo_d5_combo_eval_log.txt (22 B)
- [FILE] evo_r20_a1_adain10_log.txt (23.0 KB)
- [FILE] evo_r20_a2_long10_log.txt (1.4 MB)
- [FILE] evo_r20_evo_a1_adain10_log.txt (712.2 KB)
- [FILE] evo_r20_evo_a2_long10_log.txt (177.4 KB)
- [FILE] evo_r20_timing_results.txt (550 B)
- [DIR ] idt_baseline (170.0 MB)
- [DIR ] latent_wct_baseline ( B)
- [DIR ] musiq_d1_ll_anchor (81.6 MB)
- [DIR ] musiq_s1_sem_region (163.6 MB)
- [DIR ] musiq_s2_sem_patch (155.9 MB)
- [DIR ] musiq_s3_sem_band (154.4 MB)
- [DIR ] musiq_s4_sem_xattn (160.8 MB)
- [DIR ] musiq_s5_band_xattn (153.1 MB)
- [DIR ] musiq_s6_patch_xattn (154.6 MB)
- [DIR ] musiq_s7_dwt_energy (156.8 MB)
- [DIR ] musiq_s8_combined (156.1 MB)
- [DIR ] plugin_sd15 (692.4 MB)
- [FILE] README.md (5.0 KB)
- [DIR ] rebuttal (88.5 MB)
- [DIR ] sem_r16_k_sweep (100.8 MB)
- [DIR ] sem_r32_k_sweep (750.0 KB)
- [DIR ] sem_r4_k_sweep (93.7 MB)
- [DIR ] sem_r8_hier_mechanism (95.4 MB)
- [DIR ] sem_r8_sinkhorn_mechanism (92.5 MB)
- [DIR ] sem_r8_soft_mechanism (87.0 MB)
- [DIR ] train_taesd (313.3 MB)

### maths/

- [FILE] ANALYSIS.md (854 B)
- [FILE] CSV_STATISTICAL_APPENDIX_2026-05-16.md (7.3 KB)
- [FILE] DECISION_TREE_AND_EXPERIMENT_PLAN.md (11.9 KB)
- [FILE] FULL_EVIDENCE_ANALYSIS_2026-05-16.md (23.3 KB)
- [FILE] MODEL.md (7.1 KB)
- [FILE] REFLECTIONS.md (7.8 KB)
- [FILE] THEORY_RESET_2026-05-16.md (4.2 KB)

### probes/

- [FILE] probe_velocity_adain.py (27.4 KB)

### results/

- [FILE] _artfid_details.json (77.5 KB)
- [FILE] _artfid_summary.json (5.9 KB)
- [FILE] _cfsd_summary.json (2.6 KB)
- [FILE] _cut_p256_musiq.json (108 B)
- [FILE] _remote_map.json (2.3 KB)
- [DIR ] _teaser_p2a_top100 (254.9 MB)
- [DIR ] _teaser_top100 (167.9 MB)
- [FILE] _verify_hh.json (320 B)
- [FILE] _verify_ladder.json (309 B)
- [FILE] ablation_summary_capacity.csv (3.1 KB)
- [FILE] ablation_summary_capacity.json (5.7 KB)
- [FILE] ablation_summary_loss.csv (4.4 KB)
- [FILE] ablation_summary_loss.json (8.4 KB)
- [DIR ] cross_protocol (4.37 GB)
- [DIR ] D5-512 (7.91 GB)
- [FILE] musiq_results_r1r2r3.json (1.5 KB)
- [FILE] musiq_results_r4.json (2.5 KB)
- [FILE] musiq_verified_clean.json (96 B)
- [DIR ] ours (10.03 GB)
- [DIR ] P256 (1.12 GB)
- [FILE] phase3_minimal.csv (4.5 KB)
- [FILE] phase3_minimal.json (8.5 KB)
- [FILE] phase32_loss_new_summary.csv (3.3 KB)
- [FILE] phase32_loss_new_summary.json (5.8 KB)
- [DIR ] R5 (4.74 GB)
- [FILE] r5_musiq_baseline.json (116 B)
- [FILE] r5_musiq_sem_r8_ht008.json (113 B)
- [DIR ] R5-512 (3.91 GB)
- [DIR ] R5-WikiArt (5.55 GB)
- [FILE] README.md (2.6 KB)
- [FILE] semantic_swd_metrics_summary.json (2.0 KB)
- [FILE] semantic_swd_musiq_results.json (1.9 KB)
- [DIR ] tables (10.5 KB)
- [FILE] task2_1_attention.csv (4.5 KB)
- [FILE] task2_1_attention.json (8.5 KB)
- [FILE] task2_2_stylefilm.csv (1.7 KB)
- [FILE] task2_2_stylefilm.json (2.8 KB)
- [FILE] task2_3_endpoint.csv (3.9 KB)
- [FILE] task2_3_endpoint.json (7.2 KB)
- [FILE] task2_4_gate_init.csv (2.4 KB)
- [FILE] task2_4_gate_init.json (4.3 KB)
- [FILE] tgt_artfid_random_stability.csv (1.7 KB)
- [FILE] tgt_artfid_random_stability.json (54.7 KB)

### runs/

- [FILE] endpoint_adain_1p5.csv (229 B)
- [FILE] endpoint_adain_2p0.csv (233 B)
- [FILE] endpoint_adain_2p5.csv (235 B)
- [FILE] equiv_old.pt (50.1 KB)
- [FILE] equiv_root.pt (50.1 KB)

### scripts/

- [FILE] _audit_stylealigned_cross.py (2.5 KB)
- [FILE] _print_cross_table.py (950 B)
- [FILE] _repro_dino_eval.py (2.5 KB)
- [FILE] _scan_ablation_sources.py (2.4 KB)
- [FILE] batch_eval_all.py (17.4 KB)
- [FILE] bench_inference_latency.py (13.3 KB)
- [FILE] build_submission_pack.py (14.0 KB)
- [FILE] finalize_cross_protocol_report.py (11.0 KB)
- [FILE] pull_submission_pack_remote.bat (3.2 KB)
- [FILE] README.md (489 B)
- [FILE] recompute_aux_tables_cross.py (16.4 KB)
- [FILE] recompute_cross_protocol.py (24.7 KB)
- [FILE] recompute_pack_ablation_cross.py (7.0 KB)
- [FILE] register_other5.py (9.2 KB)
- [FILE] run_bench_inference_latency.bat (153 B)
- [FILE] run_dino_on_pack_ablations.py (2.1 KB)
- [FILE] run_submission_repro.ps1 (1.2 KB)
- [DIR ] scripts (134.5 KB)

### state/

- [FILE] baseline_main_table_on_I.md (59.4 KB)
- [DIR ] dino (5.0 MB)
- [DIR ] dino_avg (17.1 KB)
- [DIR ] model_probe (22.4 KB)
- [FILE] our_ckpts_clip_lpips.csv (26.3 KB)
- [FILE] OURS_CATEGORIZED.md (11.9 KB)
- [FILE] radar_baselines.png (650.6 KB)
- [FILE] radar_baselines_14.png (945.2 KB)
- [FILE] radar_baselines_17.png (792.8 KB)
- [FILE] radar_baselines_dino12.png (647.7 KB)
- [FILE] remote_images_inventory.md (6.5 KB)
- [FILE] remote_inventory.md (3.2 KB)
- [DIR ] semantic_swd (22.6 KB)

### tests/

- [FILE] _smoke_t26.py (4.5 KB)
- [FILE] README.md (395 B)
- [FILE] test_630_spectral_ode.py (3.2 KB)
- [FILE] test_710_inference_infra.py (1.3 KB)
- [FILE] test_72_cleanup_smoke.py (21.0 KB)
- [FILE] test_internal_dynamics.py (2.1 KB)
- [FILE] test_submission_root_layout.py (5.5 KB)
- [FILE] test_training_log_schema.py (6.9 KB)

### tools/

- [FILE] batch_eval_wfi.py (5.0 KB)
- [FILE] bench_sd15.py (4.4 KB)
- [FILE] benchmark_timing.py (12.3 KB)
- [FILE] build_other5_dataset.py (2.9 KB)
- [FILE] create_idt_other5.py (1.9 KB)
- [FILE] encode_wikiart27_sd15.py (7.7 KB)
- [FILE] fast_vae_encode_d5.py (40.2 KB)
- [FILE] infer_adain.py (21.8 KB)
- [FILE] infer_latent_wct.py (9.8 KB)
- [FILE] infer_sd_variants.py (7.8 KB)
- [FILE] infer_wct.py (13.5 KB)
- [FILE] make_r5_config.py (2.0 KB)
- [FILE] plot_pareto_scatter.py (8.4 KB)
- [FILE] preencode_d5_packed.py (4.0 KB)
- [FILE] prepare_d5_512_clean.py (2.5 KB)
- [FILE] prepare_samam_eval.py (2.3 KB)
- [FILE] taesd_encode_d5.py (4.9 KB)
- [FILE] vae_encode_p2a256.py (3.9 KB)
- [FILE] vae_encode_r5.py (5.6 KB)

### utils/

- [FILE] __init__.py (192 B)
- [FILE] artfid_metric.py (9.4 KB)
- [FILE] classify.py (23.9 KB)
- [FILE] compute_dino_metrics.py (8.9 KB)
- [FILE] dataset.py (40.4 KB)
- [FILE] diffeomorphic.py (14.7 KB)
- [FILE] inference.py (40.5 KB)
- [FILE] introstyle_eval.py (13.6 KB)
- [FILE] modern_metrics.py (17.6 KB)
- [FILE] run_evaluation.py (239.5 KB)
- [FILE] style_classifier.py (56.1 KB)
- [FILE] targetwise_artfid_summary.py (3.7 KB)
- [FILE] training.py (39.3 KB)
- [FILE] wfi.py (13.9 KB)

## 4. Root-level Code Files

Core WEAVE implementation (importable as weave package):

| File | Description |
|------|-------------|
| model.py | Main WEAVE model definition |
| flow.py | Flow network (diffeomorphic matching) |
| wavelet.py | Haar wavelet transform |
| trainer.py | Training loop and optimization |
| run.py | Main entry point (train/infer) |
| run_evaluation.py | Evaluation orchestration |
| blocks.py | Network building blocks |
| style.py | Style feature extraction |
| style_families.py | Style family definitions |
| internal_dynamics.py | Internal dynamics module |
| config_schema.py | Configuration schema (Pydantic) |
| config.json | Default model config |
| inference.json | Default inference config |
| __init__.py / __main__.py | Package init / CLI entry |

## 5. Datasets

### D5-512 (Distinct 5 styles, 512px)
- 5 style categories: Cubism, Expressionism, Pop_Art, Romanticism, Symbolism
- Test set: 5 styles x 30 images = 150 images (hold-out)
- Location: data/test/

### R5-512 (Random 5 styles, 512px)
- 5 style categories: Cubism, Expressionism, Pop_Art, Romanticism, Symbolism
- Test set: 5 styles x 30 images = 150 images
- Location: data/test/

### P2A-256 (Photo-to-Art, 256px)
- Content images (photos) + style images (artworks)
- Test set: 256px resolution
- Location: data/test_p2a/

## 6. Key Experiments (Paper-cited)

| Experiment | Location | Paper Section |
|------------|----------|---------------|
| Main checkpoint (4-epoch) | exp/dino_s_break/brk_a_ll03_10ep/ | Section 5 Main Table |
| DWT route ablation | exp/dwt_route_distinct5/ | Section 5 Ablation |
| Endpoint-only ablation | runs/ablation_endpoint_only/ | Section 5 Ablation |
| Stepwise ablation | runs/ablation_stepwise/ | Section 5 Ablation |
| B2 Qualitative | exp/b2_qualitative/ | Section 5 Qualitative |
| B3 Failure cases | exp/b3_failure_cases/ | Section 5 Analysis |
| B5 Compute cost | exp/b5_compute_cost/ | Section 5 Efficiency |
| IDT baseline | exp/idt_baseline/ | Section 5 Baselines |
| Latent-WCT baseline | exp/latent_wct_baseline/ | Supplement S9 |
| SD1.5 plugin | exp/plugin_sd15/ | Frozen SD1.5 Extension |
| DINO avg vs max | exp/dino_avg_vs_max/ | Section 5 Discussion |
| Rebuttal | exp/rebuttal/ | Reviewer response |

## 7. Paper

- Main paper: aaai2027_v4/paper.tex (compiles to paper.pdf)
- Supplement: aaai2027_v4/supplement.tex (compiles to supplement.pdf)
- Figures: aaai2027_v4/fig_*.png, aaai2027_v4/fig_*.pdf
- Bibliography: aaai2027_v4/refs.bib
- Build script: aaai2027_v4/build.ps1

## 8. Reproduction Guide

### Environment
```
pip install -r requirements.txt
```

### Training
```
python -m weave --config config.json --train
```

### Inference
```
python -m weave --config inference.json --infer
```

### Evaluation
```
python -m weave --eval --output-dir results/
```

## 9. Archive Contents (Historical Backup)

The archive/ directory contains historical experiments and data
preserved for reproducibility audit. Key subdirectories:

| Subdirectory | Content |
|--------------|---------|
| archive/top_level/72_fewshot/ | Few-shot experiments (not in paper) |
| archive/top_level/eval_cache/ | Evaluation caches (rebuildable) |
| archive/top_level/results/ | Old results snapshots |
| archive/subpack/ | Old packaging artifacts |
| archive/runs_cache/ | Run caches (rebuildable) |
| archive/runs_submission/ | Submission run checkpoints |
| archive/runs_probe_whitening*/ | Whitening probe experiments |
| archives/results_R5-WikiArt_legacy_20260721/ | Legacy R5 results |

## 10. Total Size

Total repository size: **155.48 GB** (569577 files)

---
*This inventory was auto-generated. For questions, refer to docs/ directory.*
