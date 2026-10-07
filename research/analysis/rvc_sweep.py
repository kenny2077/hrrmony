"""Model A (r3gm/villager) at -8 st: which RVC settings reproduce their clean texture?"""
import glob
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
from infer_rvc_python import BaseLoader  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "repro/vocals_excerpt.wav"
out = "analysis/rvc_sweep"
os.makedirs(out, exist_ok=True)
pth = glob.glob("models/rvc/r3gm_villager/*.pth")[0]
idx = glob.glob("models/rvc/r3gm_villager/*.index")[0]
CONF = {  # tag: (pitch_algo, index, filter, envelope_ratio, protect)
    "base": ("rmvpe", 0.75, 3, 0.25, 0.33),
    "idx0": ("rmvpe", 0.0, 3, 0.25, 0.33),
    "idx1": ("rmvpe", 1.0, 3, 0.25, 0.33),
    "prot05": ("rmvpe", 0.75, 3, 0.25, 0.5),
    "prot0": ("rmvpe", 0.75, 3, 0.25, 0.0),
    "env1": ("rmvpe", 0.75, 3, 1.0, 0.33),
    "env0": ("rmvpe", 0.75, 3, 0.0, 0.33),
    "harvest": ("harvest", 0.75, 3, 0.25, 0.33),
    "pm": ("pm", 0.75, 3, 0.25, 0.33),
    "crepe": ("crepe", 0.75, 3, 0.25, 0.33),
}
conv = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)
for tag, (algo, ir, fr, env, prot) in CONF.items():
    try:
        conv.apply_conf(tag=tag, file_model=pth, pitch_algo=algo, pitch_lvl=-8, file_index=idx,
                        index_influence=ir, respiration_median_filtering=fr, envelope_ratio=env,
                        consonant_breath_protection=prot)
        tmp = f"{out}/_in_{tag}.wav"
        shutil.copy(src, tmp)
        res = conv([tmp], [tag], overwrite=True, parallel_workers=1, type_output="wav", show_progress=False)
        shutil.move(res[0] if isinstance(res, list) else res, f"{out}/A_{tag}.wav")
        print("wrote", tag, flush=True)
    except Exception as e:  # some pitch algos may be unavailable
        print("FAILED", tag, e, flush=True)
