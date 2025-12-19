# PtuFLIM (readPTU_FLIM)

Tools for **reading PicoQuant `.ptu` TTTR scan files** and running **FLIM (TCSPC) lifetime analysis**:
- Read PTU headers + TTTR photon stream (T2/T3; MultiHarp/HydraHarp/TimeHarp/PicoHarp formats)
- Convert photon streams into **per-frame intensity + per-pixel TCSPC histograms**
- Fit IRF-convolved **multi-exponential** models (LS / ML)
- Fast **pattern matching / unmixing** of decays (CPU, and optional GPU acceleration via PyCUDA)

> The scripts `CellsegPTU_FLIM.py` and `FlavMetaFLIM.py` are **use cases** (end-to-end pipelines).  
> The reusable library code lives primarily in `PTU_ScanRead.py`, `FLIM_fitter.py`, and (optionally) `PIRLS_cu/pirls_pycuda.py`.

---

## Installation

### Conda (recommended)
An `environment.yml` is included in the repo. Create and activate:

```bash
conda env create -f environment.yml
conda activate ptu_flim
```

### Optional: PlantSeg (only needed for `CellsegPTU_FLIM.py`)
`CellsegPTU_FLIM.py` uses PlantSeg for deep-learning segmentation. Install PlantSeg in your environment if you plan to run that script.

### Optional: CUDA / PyCUDA (GPU acceleration for PIRLS)
If you have an NVIDIA GPU and want faster pattern matching (PIRLS), install PyCUDA and a matching CUDA toolkit/driver stack.

---

## Typical workflow (high level)

1. **Read the PTU scan stream** into arrays (`PTU_ScanRead`)
2. **Build per-frame images / per-pixel TCSPC histograms** (`Process_Frame`)
3. **Estimate an IRF** from a measured TCSPC IRF acquisition (`Calc_mIRF`) or use your own IRF
4. **Fit lifetimes** (`FluoFit`) and/or **unmix decays** (`PatternMatchIm`)

---

## Quick start

### Read a PTU scan
```python
from PTU_ScanRead import PTU_ScanRead

head, im_sync, im_tcspc, im_chan, im_line, im_col, im_frame = PTU_ScanRead("your_file.ptu")
print(head["ImgHdr_PixX"], head["ImgHdr_PixY"], head["ImgHdr_MaxFrames"])
```

### Build per-frame per-pixel TCSPC histograms
```python
import numpy as np
from PTU_ScanRead import Process_Frame

resolution_ns = 0.2   # output TCSPC bin width in ns (will be clamped to the hardware bin if smaller)
cnum = head.get("PIENumPIEWindows", 1)  # PIE windows (if present)

frame_id = 0
idx = np.where(im_frame == frame_id)[0]

tag, tau_sigma, tcspc_pix = Process_Frame(
    im_sync[idx], im_col[idx], im_line[idx], im_chan[idx], im_tcspc[idx],
    head, cnum=cnum, resolution=resolution_ns
)

# tag: (nx, ny, n_det, cnum)  summed counts per pixel
# tau_sigma: (nx, ny, n_det, cnum)  sigma from TCSPC moment (quick “spread” estimate)
# tcspc_pix: (nx, ny, n_bins, n_det*cnum) per-pixel TCSPC histograms
```

### Fit a single decay curve (multi-exponential)
```python
import numpy as np
from FLIM_fitter import Calc_mIRF, FluoFit

# Example: use channel 0 / PIE window 0 for a single pixel
y = tcspc_pix[10, 10, :, 0].astype(float)

# Build / load an IRF histogram (here: example from a measured IRF TCSPC histogram)
# tcspc_irf should be a 1D histogram aligned to y (same number of bins)
# irf = Calc_mIRF(head, tcspc_irf)
# For demonstration, reuse a normalized copy of y as a placeholder IRF:
irf = y / (y.sum() + 1e-12)

# Pulse period in ns (one PIE window)
p_ns = int(np.floor(head["MeasDesc_GlobalResolution"] * 1e9 / cnum + 0.5))

# Initial lifetime guesses in ns (choose based on your sample)
tau0 = np.array([0.5, 2.0, 6.0])

taufit, A, coeffs, zfit, patterns, shift, irf_shifted, t_ns, chi = FluoFit(
    irf, y, p_ns, resolution_ns, tau=tau0, flag_ml=False, plt_flag=0
)
```

### Pattern matching (unmixing) of per-pixel decays
Once you have a set of decay “patterns” (columns of `patterns` from `FluoFit`), you can estimate per-pixel amplitudes:

```python
from FLIM_fitter import PatternMatchIm

# y_im: (nx, ny, t) — e.g., tcspc_pix[:, :, :, 0]
# patterns: (t, n_basis) from FluoFit
C, Z = PatternMatchIm(tcspc_pix[:, :, :, 0], patterns, mode="PIRLS")  # or "Nonneg" / "Default"
# C: (nx, ny, n_basis) amplitude maps
# Z: (nx, ny, t) reconstructed decays
```

---

## Module reference

## `PTU_ScanRead.py`

### `PTUreader`
Low-level PTU parser / TTTR reader.

- `PTUreader(filename, print_header_data=False)`  
  Loads the file into memory, parses the header into `self.head`.

- `get_photon_chunk(start_record, n_records, head=None)`  
  Returns a chunk of decoded TTTR records as `(sync, tcspc, channel, special, num, loc)`.

Internally, `_ptu_read_raw_data()` implements record-type decoding for supported PicoQuant TTTR formats.

### `PTU_ScanRead(filename, cnum=1, plt_flag=False)`
Reads a scan acquisition and returns arrays suitable for downstream reconstruction.

**Returns**
- `head`: header dictionary
- `im_sync`: sync index per photon
- `im_tcspc`: TCSPC bin index per photon
- `im_chan`: detector/input channel per photon
- `im_line`: line index per photon
- `im_col`: column index per photon
- `im_frame`: frame index per photon

### `Process_Frame(im_sync, im_col, im_line, im_chan, im_tcspc, head, cnum=1, resolution=0.2)`
Builds per-pixel TCSPC histograms for one frame.

**Returns**
- `tag (nx, ny, n_det, cnum)`: summed counts per pixel (intensity)
- `tau (nx, ny, n_det, cnum)`: moment-based **spread** estimate from the histogram  
  σ = sqrt(E[t²] - (E[t])²) (computed using the binned histogram)
- `tcspc_pix (nx, ny, n_bins, n_det*cnum)`: per-pixel TCSPC histograms

### `Harp_TCPSC(filename, resolution=None, deadtime=None, photons=None)`
Convenience reader for “global” TCSPC histograms (aggregated over all pixels), returning:
- `tcspcdata (n_bins, n_det)`: histograms per detection channel
- `binT (n_bins,)`: time-bin index (convert to time using your chosen resolution)
- `head`: header dictionary

### Histogram utilities
- `mHist(x, xv=None)` → `(h, xv)`
- `mHist2(x, y, xv=None, yv=None)` → `(h, xv, yv)`
- `mHist3(x, y, z, xv=None, yv=None, zv=None)` → `(h, xv, yv, zv)`
- `mHist4(x, y, z, t, xv=None, yv=None, zv=None, tv=None)` → `(h, xv, yv, zv, tv)`

### Display utility
- `cim(x, ...)`  
  MATLAB-like “cim” visualization helper (returns the matplotlib image handle).

---

## `FLIM_fitter.py`

### Core building blocks
- `Convol(irf, x)`  
  FFT-based convolution of an IRF with one or many decay curves.

- `IRF_Fun(p, t, pic=None)`  
  Parametric IRF model (Gaussian peak + tails) returning a normalized IRF.

- `TCSPC_Fun(p, t, y=None, para=None)`  
  Builds an IRF-convolved multi-exponential design matrix and (optionally) fits amplitudes to data using NNLS.  
  Returns `(err, c, zz, z)` where `z = zz @ c`.

### IRF helper
- `Calc_mIRF(head, tcspc)`  
  Creates a *measured* IRF (mIRF) from an IRF TCSPC histogram.

### Optimization / fitting
- `LSFit(param, y, irf, p, plt_flag=None)`  
  Least-squares objective for optimizing IRF shift + lifetimes.

- `MLFit(param, y, irf, p, plt_flag=None)`  
  Poisson negative log-likelihood objective for optimizing IRF shift + lifetimes.

- `FluoFit(irf, y, p, dt, tau=None, lim=None, flag_ml=False, plt_flag=1)`  
  High-level fitter. Typical use: provide an IRF histogram, a decay `y`, pulse period `p` (in the same units as `dt`), and initial lifetime guesses `tau`.  
  Returns:
  - `taufit`: fitted lifetimes (same units as `dt`)
  - `A`: fitted amplitudes (including background term)
  - `c`: coefficients (internal)
  - `zfit`: fitted curve
  - `patterns`: basis curves (background + each convolved component)
  - `offset`: fitted IRF shift
  - `irs`: shifted IRF
  - `t`: time axis
  - `chi`: reduced χ²-like metric

- `DistFluoFit(y, p, dt, irf=None, shift=(-10,10), flag=0, bild=None, N=100, scattering=True)`  
  Distributed lifetime fitting / initialization helper (regularized linear regression over candidate lifetimes).  
  Returns `(cx, tau, offset, csh, z, t, err)`.

### Pattern matching (unmixing)
- `PatternMatchIm(y, M, mode="Default")`  
  Estimates coefficient maps `C` such that `y ≈ M @ C` for every pixel.  
  `mode` options:
  - `"Default"`: unconstrained least squares
  - `"Nonneg"`: non-negative fit (SciPy NNLS / bounded LS)
  - `"PIRLS"`: Poisson IRLS with non-negativity (recommended for photon-counting decays)

Returns `(C, Z)` where `Z` is the reconstruction.

### PIRLS solvers (CPU + optional GPU)
- `PIRLSnonneg(x, y, max_num_iter=10)`  
  Single-curve Poisson IRLS NNLS solver.

- `PIRLSnonneg_batch(M, Y, max_num_iter=10)`  
  Batch PIRLS for many curves (columns of `Y`).  
  If PyCUDA is available, it will try to accelerate via `pirls_batch_cuda` (see below) for problems with:
  - `n_basis <= 64`
  - `n_samples <= 2048`

---

## `PIRLS_cu/pirls_pycuda.py` (GPU acceleration)

This module provides a **PyCUDA** implementation of batch PIRLS NNLS used by `PIRLSnonneg_batch` / `PatternMatchIm(mode="PIRLS")`.

### Entry points
- `pirls_batch_cuda(M, Y, max_iter=10)`  
  Convenience wrapper. Inputs are float32 arrays:
  - `M`: (n_samples, n_basis)
  - `Y`: (n_samples, n_pixels)
  Returns `Beta`: (n_basis, n_pixels)

- `run_pirls_pycuda(M, Y, n_samples, n_basis, n_pixels, max_iter=10)`  
  Lower-level runner.

### Notes
- Kernel constants limit sizes (currently `MAX_BASIS=64`, `MAX_SAMPLES=2048`).
- If CUDA compilation fails, code falls back to CPU automatically.

### Validation
`test_PIRLS.py` compares SciPy NNLS / bounded LS, CPU PIRLS, and (if available) CUDA PIRLS on synthetic decays.

---

## Lifetime “summary” metrics (merged from `LIFETIME_METHODS_README.md`)

When you fit multiple components (lifetimes τᵢ with amplitudes Aᵢ), you may want a single representative number per pixel. Common choices:

### 1) Intensity-weighted mean lifetime (normalized amplitude-weighted)
Used in `FlavMetaFLIM.py`:
\[
	au_{avg} = \frac{\sum_i A_i \, \tau_i}{\sum_i A_i}
\]

### 2) Rate-averaged lifetime (harmonic mean)
Also used in `FlavMetaFLIM.py`:
\[
	au_{rate} = \frac{\sum_i A_i}{\sum_i A_i/\tau_i}
\]
This emphasizes faster components more strongly than the arithmetic mean.

### 3) Moment / variance-based spread from the histogram
Computed in `Process_Frame()` as a quick per-pixel estimate of the TCSPC temporal spread:
\[
\sigma = \sqrt{E[t^2] - (E[t])^2}
\]
This is not the same as a multi-exponential fit, but can be useful as a fast summary statistic.

---

## Use cases

### `FlavMetaFLIM.py`
Metabolic (FAD) FLIM pipeline:
- Reads PTU
- Generates per-pixel decays
- Fits lifetimes (`FluoFit`)
- Computes intensity- and rate-averaged lifetime images
- Exports images + summary tables

### `CellsegPTU_FLIM.py`
Cell segmentation + FLIM pipeline:
- Reads PTU
- Runs PlantSeg segmentation (U-Net + mutex watershed)
- Fits per-cell/per-pixel lifetimes and visualizes results

---

## License
Third-party code: PTU_ScanRead.py includes components derived from
SumeetRohilla/readPTU_FLIM, licensed under the MIT License.

