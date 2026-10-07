"""Matched-filter search for Minecraft samples inside audio stems (GPU).

For every template clip (villager-family voices, player attack / damage / note-block / random SFX)
and every playback rate 2**(k/24), k=-24..24 (Minecraft re-pitches by resampling), compute

  A) waveform normalised cross-correlation (exact sample reuse, survives gain/mixing)
  B) log-mel patch Pearson correlation (survives EQ/compression/phase changes)

against each target stem. Controls: the original Rick Astley mix (no Minecraft content) and
time-reversed templates (same spectrum, wrong temporal shape).

    .venv-conv/bin/python analysis/hit_scan.py
writes analysis/hit_scan.json
"""

import glob
import json
import os

import librosa
import numpy as np
import torch

SR = 16000
DEV = "cuda"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = [2 ** (k / 24) for k in range(-24, 25)]
MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=64, fmin=60, fmax=7600)
HOP = 160  # 10 ms


def load(p, offset=0.0, dur=None):
    y, _ = librosa.load(p, sr=SR, mono=True, offset=offset, duration=dur)
    return y.astype(np.float32)


def logmel(y):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)) ** 2
    return np.log(MEL @ S + 1e-8).astype(np.float32)


class Target:
    def __init__(self, y):
        self.n = len(y)
        self.x = torch.tensor(y, device=DEV, dtype=torch.float64)
        self.cs = torch.cumsum(torch.cat([torch.zeros(1, device=DEV, dtype=torch.float64), self.x ** 2]), 0)
        self.nfft = 1 << int(np.ceil(np.log2(self.n + 48000)))
        self.X = torch.fft.rfft(self.x, n=self.nfft)
        M = torch.tensor(logmel(y), device=DEV)  # (64, T)
        self.M = M
        self.Mcs = torch.cumsum(torch.cat([torch.zeros(64, 1, device=DEV), M], 1), 1)
        self.Mcs2 = torch.cumsum(torch.cat([torch.zeros(64, 1, device=DEV), M ** 2], 1), 1)

    def ncc_wave(self, t):
        m = len(t)
        tt = torch.tensor(t, device=DEV, dtype=torch.float64)
        tt = tt - tt.mean()
        T = torch.fft.rfft(torch.flip(tt, [0]), n=self.nfft)
        num = torch.fft.irfft(self.X * T, n=self.nfft)[m - 1:self.n]
        e = self.cs[m:] - self.cs[:-m]
        den = torch.sqrt(torch.clamp(e, min=1e-9)) * torch.linalg.norm(tt)
        r = num[:len(e)] / den
        # ignore near-silent windows (energy below -50 dB of the stem's loud level)
        loud = e / m > (self.cs[-1] / self.n) * 1e-3
        return torch.where(loud, r, torch.zeros_like(r))

    def ncc_mel(self, tm):
        tm = torch.tensor(tm, device=DEV)
        L = tm.shape[1]
        if L >= self.M.shape[1]:
            return torch.zeros(1, device=DEV)
        tz = tm - tm.mean()
        tn = torch.linalg.norm(tz)
        num = torch.nn.functional.conv1d(self.M[None], tz[None]).flatten()  # sum M*tz
        s1 = (self.Mcs[:, L:] - self.Mcs[:, :-L]).sum(0)
        s2 = (self.Mcs2[:, L:] - self.Mcs2[:, :-L]).sum(0)
        cnt = 64 * L
        var = torch.clamp(s2 - s1 ** 2 / cnt, min=1e-6)
        return num / (torch.sqrt(var) * tn)


def peaks(r, hop_s, k=5, sep=0.25):
    r = r.clone()
    out = []
    w = max(1, int(sep / hop_s))
    for _ in range(k):
        i = int(torch.argmax(r))
        v = float(r[i])
        if v <= 0:
            break
        out.append((round(i * hop_s, 3), round(v, 3)))
        r[max(0, i - w):i + w] = 0
    return out


def main():
    tpl_paths = sorted(glob.glob(os.path.join(HERE, "ref/library/*.ogg")) +
                       glob.glob(os.path.join(HERE, "ref/sfx/*.ogg")))
    tpl = {}
    for p in tpl_paths:
        name = os.path.basename(p)[:-4]
        if name.startswith("mob__villager__"):
            continue  # duplicates of ref/library/villager__*
        y = load(p)
        y, _ = librosa.effects.trim(y, top_db=40)
        if len(y) < 0.04 * SR:
            continue
        tpl[name] = y[: int(1.5 * SR)]
    targets = {
        "their_vocal": os.path.join(HERE, "work/reference_full/vocals.wav"),
        "their_instr": os.path.join(HERE, "work/reference_full/instrumental.wav"),
        "their_mix": os.path.join(HERE, "work/reference_full/clip.wav"),
        "orig_mix(control)": os.path.join(HERE, "input/rickroll.mp3"),
    }
    res = {}
    for tname, tp in targets.items():
        T = Target(load(tp))
        res[tname] = {}
        for name, y in tpl.items():
            best = dict(wave=(-1, None, None), mel=(-1, None, None), wave_rev=-1, mel_rev=-1)
            allpk = []
            for rate in RATES:
                # playback at `rate` = resample so the clip is 1/rate as long (and pitched by rate)
                yr = librosa.resample(y, orig_sr=SR, target_sr=int(round(SR / rate)))
                if len(yr) < 400:
                    continue
                rw = T.ncc_wave(yr)
                pk = peaks(rw, 1 / SR, k=3)
                if pk and pk[0][1] > best["wave"][0]:
                    best["wave"] = (pk[0][1], round(rate, 3), pk[0][0])
                allpk += [(p[0], p[1], round(rate, 3)) for p in pk]
                rr = T.ncc_wave(yr[::-1].copy())
                best["wave_rev"] = max(best["wave_rev"], float(rr.max()))
                tm = logmel(yr)
                if tm.shape[1] >= 4:
                    rm = T.ncc_mel(tm)
                    pm = peaks(rm, HOP / SR, k=1)
                    if pm and pm[0][1] > best["mel"][0]:
                        best["mel"] = (pm[0][1], round(rate, 3), pm[0][0])
                    best["mel_rev"] = max(best["mel_rev"], float(T.ncc_mel(tm[:, ::-1].copy()).max()))
            allpk.sort(key=lambda p: -p[1])
            best["top_wave_peaks"] = allpk[:8]
            res[tname][name] = best
        print("done", tname, flush=True)
    json.dump(res, open(os.path.join(HERE, "analysis/hit_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
