{
  description = "Define development dependencies.";


  inputs = {
    # Which Nix upstream package branch to track
    nixpkgs.url = "nixpkgs/nixos-24.11";
    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";
    services-flake.url = "github:juspay/services-flake";
  };

  # What results we're going to expose
  outputs = { nixpkgs, process-compose-flake, services-flake, ... }:
    let

      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {

        pkgs = import nixpkgs { 
          inherit system; 
          config.allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) [
            "cuda_nvcc"
          ];
        };
        

        servicesMod = (import process-compose-flake.lib { inherit pkgs; }).evalModules {
          modules = [
            services-flake.processComposeModules.default
            {
              services.ollama."ollama1" = {
                enable = true;
                acceleration = "rocm";
              };
            }
          ];
        };
      });

    in {
      packages = forAllSystems ({ servicesMod, ... }: {
        default = servicesMod.config.outputs.package;
      });


      # Declare what packages we need as a record. The use as a record is
      # needed because, without it, the data contained within can't be
      # referenced in other parts of this file.
      devShells = forAllSystems ({pkgs, servicesMod}: {
        default = pkgs.mkShell rec {
          packages = with pkgs; [
            python311Full
            # (poetry.override { python3 = pkgs.python311; })
            python311Packages.distlib
            python311Packages.cython
            python311Packages.setuptools
            python311Packages.setuptoolsBuildHook
            python311Packages.wheel
            direnv
            pkg-config
            cmake
            blas
            lapack
            gcc_multi 
            gccMultiStdenv
            ruff
            ninja
            gfortran
            meson
            glibc_multi
            ollama-rocm
            openblas
            cudaPackages.cuda_nvcc
            zlib
            # NOTE: Put additional packages you need in this array. Packages may be found by looking them up in
            # https://search.nixos.org/packages
          ];# ++ (with lib; 
          #  filter (x: x ? outPath) (
          #    builtins.attrValues (
          #      filterAttrs (name: _: all (term: !(hasInfix term (toLower name))) [ 
          #        "auto-add-cuda-compat-runpath-hook"
          #        "autoAddCudaCompatRunpath"
          #      ]) 
          #      #(
          #      #  filterAttrs (isBroken: marked_broken: !(marked_broken))
          #        cudaPackages
          #      #)
          #    )
          #  )
          #);

          # Getting the library paths needed for Python to be put into
          # LD_LIBRARY_PATH
          pythonldlibpath = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.lib.makeLibraryPath packages}:$NIX_LD_LIBRARY_PATH";

          # Run the following on the shell, which builds up LD_LIBRARY_PATH.
          shellHook = ''
              export LD_LIBRARY_PATH="${pythonldlibpath}"
          '';
        };
      });
    };
}
