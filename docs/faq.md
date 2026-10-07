# FAQ and troubleshooting

**`ffmpeg/ffprobe not found`**
Install ffmpeg:
- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt install ffmpeg`
- Windows: `winget install ffmpeg`
- Without admin rights: download a static build and put it on your `PATH`.

Then run `hrrmony doctor`.

**It runs on CPU even though I have an NVIDIA GPU.**
`pip` installed a CPU-only torch. Reinstall torch from the PyTorch index that matches your driver, for example:
```bash
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Then check that `hrrmony doctor` shows `CUDA …`.

**The first run is slow or seems to hang.**
On first use it downloads the separator (about 870 MB), the villager voice (about 65 MB), the pitch model (about 180 MB) and the HuBERT encoder. Later runs reuse the cache in `~/.cache/hrrmony`.

**The villager sounds off key.**
That's the `classic` shift (−8 semitones). It reproduces the viral covers, which sit a major third away from the backing track. Use `--shift in-key` (−12) to stay on pitch.

**The wrong part of the song was picked.**
Pass `--start <seconds>`, for example `--start 43`.

**Rap or very fast songs sound mushy.**
Voice conversion follows the singer. Dense rap gives dense villager noise. Melodic choruses work best.

**`ImportError: HubertModel` / transformers errors.**
The RVC encoder needs `transformers<4.50`. The extras pin this. If another package upgraded it, run `pip install "transformers>=4.40,<4.50"`.

**Can I publish the covers?**
Check the song's rights and the voice model's terms first. The default villager model is a community upload ([r3gm/villager](https://huggingface.co/r3gm/villager)). Minecraft is a trademark of Mojang/Microsoft, and this project is not affiliated with them.
