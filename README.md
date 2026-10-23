# splat-3dhuman docker image

Archived on 2026-10-23. The splat project (a photoreal moving 3D human) stopped, so this image is no longer updated. The project notes and scripts are in the `splat` project in Brian's tsbasics workspace.

Baked CUDA env for the splat project's 3D-human / diffusion work on vast.ai RTX 4090 (sm_89), so a
fresh box is ready in ~2-4 min instead of a 15-25 min from-source build.

Image: `ghcr.io/briantanabe/splat-3dhuman:cu121`

Bakes the slow/fragile parts: torch 2.4.1+cu121, **pytorch3d** (built `--no-build-isolation`,
sm_89), nvdiffrast, gaussian rasterizers, trimesh/pymeshlab/xatlas/rembg/open3d, diffusers 0.31 +
transformers + gfpgan/insightface (basicsr `functional_tensor` patched). Model-specific pins (e.g.
PSHuman's diffusers) install at runtime in <1 min.

## Build
Push to `main` (or run the workflow manually) → GitHub Actions builds on a CPU runner
(`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9+PTX` compiles kernels with no GPU) and pushes to GHCR.
After the first build, set the GHCR package visibility to **Public** so vast needs no registry login.

## Boot a box
```
vastai create instance <offer> --image ghcr.io/briantanabe/splat-3dhuman:cu121 --disk 120 --ssh --direct
```

No license-gated weights are baked in; hydrate SMPL-X/checkpoints at runtime.
