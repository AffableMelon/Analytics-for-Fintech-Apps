{
  description = "PostgreSQL Support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

      in
      {
        devShells.default = pkgs.mkShell {
          name = "flask-devshell";
          buildInputs = [
            pkgs.docker
            pkgs.docker-compose
          ];
          shellHook = ''
                      echo " Docker: ${pkgs.docker.version}"
            					echo " Docker-Compose: ${pkgs.docker-compose.version}"
          '';
        };
      }
    );
}
