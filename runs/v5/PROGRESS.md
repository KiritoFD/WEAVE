[2026-11-07 07:14:04] C1_train_full START
[2026-11-07 07:29:13] C1_train_full START
[2026-11-07 09:27:19] C1_train_full END rc=0 elapsed=7086s log=runs/v5/route_c/logs/train_full.log
[2026-11-07 09:27:19] C2_noise_check START
[2026-11-07 09:27:52] C2_noise_check END rc=0 elapsed=33s log=runs/v5/route_c/logs/check.log
[2026-11-07 09:27:52] C2 DECISION pure-noise prior failed -> lowmid denoising fallback
[2026-11-07 09:27:52] C2b_train_lowmid START
[2026-11-07 10:14:46] C2b_train_lowmid END rc=0 elapsed=2814s log=runs/v5/route_c/logs/train_lowmid.log
[2026-11-07 10:14:46] C2c_lowmid_check START
[2026-11-07 10:15:21] C2c_lowmid_check END rc=0 elapsed=35s log=runs/v5/route_c/logs/check_lowmid.log
[2026-11-07 10:15:21] C3_quick_sweep START
[2026-11-07 10:18:58] C3_quick_sweep END rc=0 elapsed=217s log=runs/v5/route_c/logs/quick.log
[2026-11-07 10:18:58] C3 DECISION best_config={"ckpt": "runs/v5/route_c/flow_lowmid.pt", "tau0": 0.75, "guide": true, "seed": 42}
[2026-11-07 10:18:58] C4_final_750 START
[2026-11-07 10:19:07] C4_final_750 END rc=1 elapsed=9s log=runs/v5/route_c/logs/final.log
Traceback (most recent call last):
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 798, in <module>
    main()
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 794, in main
    args.func(args)
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 700, in stage_final
    results[tag] = render_cell(args, model, cfg, cell, seed, device)
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 589, in render_cell
    z_src, src_paths, requests, style_meta = source_latents(device, max_src=30)
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 232, in source_latents
    imgs = torch.stack([_load_eval_image_tensor(p, size=512) for p in src_paths[s:s + 8]]).to(device)
  File "/mnt/i/Latent_Style/WEAVE/tools/v5/route_c.py", line 232, in <listcomp>
    imgs = torch.stack([_load_eval_image_tensor(p, size=512) for p in src_paths[s:s + 8]]).to(device)
  File "/mnt/i/Latent_Style/WEAVE/utils/run_evaluation.py", line 1813, in _load_eval_image_tensor
    img = Image.open(path).convert("RGB").resize((size, size))
  File "/usr/local/lib/python3.10/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'Early_Renaissance'
[2026-11-07 10:19:07] C4 FATAL
[2026-11-07 10:33:06] RESUME using ckpt=runs/v5/route_c/flow_lowmid.pt config={"ckpt": "runs/v5/route_c/flow_lowmid.pt", "tau0": 0.75, "guide": true, "seed": 42}
[2026-11-07 10:33:06] RESUME C4_final_750 START
[2026-11-07 10:43:51] RESUME C4_final_750 END rc=0 elapsed=645s log=runs/v5/route_c/logs/final_resume.log
[2026-11-07 10:43:51] RESUME C5_summary START
[2026-11-07 10:43:51] RESUME C5_summary END rc=0 elapsed=0s log=runs/v5/route_c/logs/summary_resume.log
[2026-11-07 10:43:51] V5 ROUTE C ALL DONE (RESUMED)
