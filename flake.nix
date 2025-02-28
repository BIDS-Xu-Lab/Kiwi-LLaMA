{
  description = "Define development dependencies.";

  inputs = {
    # Which Nix upstream package branch to track
    nixpkgs.url = "nixpkgs/nixos-unstable";
    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";
    services-flake.url = "github:juspay/services-flake";
  };

  # What results we're going to expose
  outputs = { nixpkgs, process-compose-flake, services-flake, ... }:
    let

      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {
        pkgs = import nixpkgs { inherit system; };

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
            python3Full
            (poetry.override { python3 = pkgs.python312; })
            direnv
            gcc-unwrapped
            stdenv
            ruff
            ollama-rocm
            # NOTE: Put additional packages you need in this array. Packages may be found by looking them up in
            # https://search.nixos.org/packages
          ];

          # Getting the library paths needed for Python to be put into
          # LD_LIBRARY_PATH
          pythonldlibpath = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.lib.makeLibraryPath packages}";

          # Run the following on the shell, which builds up LD_LIBRARY_PATH.
          shellHook = ''
              export LD_LIBRARY_PATH="${pythonldlibpath}"
          '';
        };
      });
    };
}
