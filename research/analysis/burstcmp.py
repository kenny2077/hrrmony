import sys; sys.path.insert(0, "analysis"); import bursts_rvc as R
G = R.groups()
for nm, p, off in [("theirs", "work/reference_full/vocals.wav", 29.6),
                   ("repro -8", "out/rvc/rickroll_29.5s_villager_rvc-8_vocal.wav", 0),
                   ("repro -12", "out/rvc/rickroll_29.5s_villager_rvc-12_vocal.wav", 0),
                   ("ours v4", "out/final_v4/rickroll_villager_hum_only.wav", 0)]:
    r = R.per_burst(p, off, 31, G)
    print(f"{nm:10s} hit-like bursts/min {r['bursts_per_min']:5.1f}  nearest {r['nearest']}")
