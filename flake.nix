{
  description = "Python dev environment for math_agent_vikhr";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311; # Можно заменить на нужную версию
        python-with-deps = python.withPackages (ps: with ps; [
          # Минимальный набор, остальные подтянет pip install
          pip
          # Можно добавить сюда явно нужные пакеты из nixpkgs, если нужно
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ python-with-deps ];
          shellHook = ''
            export VIRTUAL_ENV="$PWD/venv"
            if [ ! -d "$VIRTUAL_ENV" ]; then
              ${python.interpreter} -m venv "$VIRTUAL_ENV"
            fi
            source "$VIRTUAL_ENV/bin/activate"
            pip install --upgrade pip
            pip install -r requirements.txt
          '';
        };
      }
    );
}