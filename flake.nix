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
        python = pkgs.python311;
        python-with-deps = python.withPackages (ps: with ps; [
          annotated-types
          anyio
          brotli
          certifi
          charset-normalizer
          distro
          filelock
          fsspec
          h11
          httpcore
          httpx
          huggingface-hub
          idna
          jinja2
          markdown-it-py
          markupsafe
          mdurl
          openai
          packaging
          pillow
          pydantic
          pydantic-core
          pygments
          python-dotenv
          pyyaml
          regex
          requests
          rich
          sniffio
          tenacity
          tiktoken
          tqdm
          urllib3
          # smolagents и jiter, hf-xet — через pip, их нет в nixpkgs
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ python-with-deps 
            # Lean 3 нужен для miniF2F proof verification
            pkgs.lean3
          ];
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