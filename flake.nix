{
  description = "Coral Fish Pipeline — uv-managed dev shell (FHS env with CUDA + graphics libs for prebuilt wheels)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # cudatoolkit and the NVIDIA driver libs are unfree.
          config.allowUnfree = true;
        };

        # Libraries the prebuilt pip/uv wheels dynamically link against.
        # opencv-python needs the X/graphics stack; torch's CUDA wheel bundles
        # its own cuDNN/cuBLAS/NCCL but still needs libcuda.so from the host
        # driver (exposed below via /run/opengl-driver/lib).
        fhsLibs = pkgs: (with pkgs; [
          # uv + python toolchain
          uv
          python311

          # Hugging Face CLI (`hf` / `huggingface-cli`) for `hf auth login` and
          # downloading the gated SAM3 / BioCLIP model repos. Standalone wrapper,
          # independent of the project .venv.
          python3Packages.huggingface-hub

          # C/C++ build toolchain — for any sdist that has no wheel for the
          # active interpreter (e.g. SAM3's pinned numpy on newer Pythons).
          stdenv.cc
          stdenv.cc.cc.lib
          gcc
          gnumake
          cmake
          ninja
          pkg-config
          gfortran # BLAS/LAPACK source builds (numpy/scipy)

          # Core runtime libs
          zlib
          glib

          # CUDA userspace (driver's libcuda.so comes from the host, see profile).
          # NOTE: do NOT add cudaPackages.cudnn here. The torch CUDA wheel bundles
          # its own cuDNN (see .venv/.../nvidia/cudnn/lib). buildFHSEnv merges all
          # targetPkgs libs into /usr/lib, so a nix cuDNN of a different build
          # shadows torch's sublibraries by SONAME and triggers
          # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH at conv2d time.
          cudatoolkit

          # OpenCV / OpenGL / X11 stack (fixes `libxcb.so.1: cannot open ...`)
          libGL
          libGLU
          libx11
          libxext
          libxcb
          libxrender
          libxrandr
          libxi
          libsm
          libice
          libxkbcommon
          fontconfig
          freetype

          # Common CLI niceties inside the sandbox
          git
          bashInteractive
          which
        ]);

        fhs = pkgs.buildFHSEnv {
          name = "coral-fish-dev";
          targetPkgs = fhsLibs;
          # Make CUDA libs and the host NVIDIA driver discoverable, and set
          # sensible uv/HF defaults. /run/opengl-driver/lib holds libcuda.so.1.
          profile = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
            export HF_HOME="''${HF_HOME:-$PWD/models/hf_cache}"
            # Pin the project interpreter; 3.11 has wheels for every dependency
            # (including SAM3's older numpy), so nothing has to build from sdist.
            export UV_PYTHON=3.11
            echo "coral-fish-dev (FHS + CUDA). Run 'uv sync' to set up .venv, then 'uv run pytest tests -q'."
          '';
          runScript = "bash";
        };
      in
      {
        devShells.default = fhs.env;
        # `nix run` also drops you straight into the FHS shell.
        packages.default = fhs;
        apps.default = {
          type = "app";
          program = "${fhs}/bin/coral-fish-dev";
        };
      });
}
