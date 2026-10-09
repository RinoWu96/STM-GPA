# STM-GPA

An interactive desktop tool for **lattice-strain mapping of STM topographs** using geometric phase analysis (GPA).
It loads Nanonis `.sxm` images and produces strain maps (εxx, εyy, εxy), lattice rotation and dilatation.

**Related STM/STS data-processing tools:**

- [STS-map](https://github.com/RinoWu96/STS-map): STS line-scan analysis: dI/dV maps, CBM/VBM detection and band gap along the line
- [BandGap](https://github.com/RinoWu96/BandGap): band gap extraction from single STS spectra
- **STM-GPA** (this repo): lattice-strain mapping of STM topographs with geometric phase analysis

## Features

- Loads Nanonis `.sxm` files; choose the channel and scan direction (forward / backward / both)
- Two analysis methods: **GPA** (Fourier-space phase analysis) and **Atomic** (real-space lattice positions)
- Guided step-by-step workflow, from Bragg peak picking to the final strain maps
- Sub-pixel fitted Bragg peaks and FFT unit-cell alignment to the known lattice (a, b, angle)
- Strain is measured relative to user-selected **strain-free reference patches**, with optional offset or drift correction
- Outputs εxx, εyy, εxy, rotation and dilatation maps, line profiles and advanced GPA diagnostics
- Saves numerical results to `.npz` for further analysis

> **Note:** strain is reported relative to the selected reference region. In a single STM topograph, scanner calibration, thermal drift, creep and affine image distortion cannot be reliably separated from a uniform lattice strain, so absolute values should be interpreted with care.

## Download (Windows)

No Python needed: download `strain.exe` from the [Releases](https://github.com/RinoWu96/STM-GPA/releases) page and double-click it.

## Run from source

Requires Python 3.9+.

```bash
git clone https://github.com/RinoWu96/STM-GPA.git
cd STM-GPA
pip install -r requirements.txt
python strain.py
```

### Build the executable yourself

```bash
pip install pyinstaller
pyinstaller strain.spec
# output: dist/strain.exe
```

## Workflow

1. **Load SXM**: open a Nanonis `.sxm` file and set the channel, scan direction and lattice constants (a, b, angle).
2. **Select analysis region**: drag a box on the image.
3. **Pick two Bragg peaks** in the FFT (or use the fitted sub-pixel version).
4. **Align FFT unit cell** to the expected lattice.
5. **Select strain-free patches** to use as the reference.
6. **Calculate selected strain**: view the strain maps and line profiles, then save the results.

## Citation

If this tool helps your research, please cite:

```
Rino. STM-GPA: geometric phase analysis of lattice strain in STM images. GitHub, 2026. https://github.com/RinoWu96/STM-GPA
```

## Contact

Rino
