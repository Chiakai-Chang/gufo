{
  lib,
  fetchFromGitHub,
  python313Packages,
}:

python313Packages.buildPythonPackage rec {
  pname = "hyperloom-inference-optimizer";
  version = "1.0.0b2-unstable-2026-08-19";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "AMD-AGI";
    repo = "Hyperloom";
    rev = "c92784cbf1c62652a75c751063c52ffecce9a909";
    hash = "sha256-nODzWaiO5uxjD+uhe4VyXb738chOIPG4afC65DVH9lk=";
  };

  build-system = with python313Packages; [
    setuptools
    wheel
  ];

  dependencies = with python313Packages; [
    cachetools
    claude-agent-sdk
    httpx
    markdownify
    openai
    packaging
    pyyaml
  ];

  # Upstream's full test suite pulls in serving frameworks and GPU tooling.
  # Keep this development package focused on the installable libraries and
  # command-line tools; validate end-to-end operation on the dedicated host.
  doCheck = false;

  pythonImportsCheck = [ "hyperloom" ];

  meta = with lib; {
    description = "Agentic LLM inference optimization runtime for AMD GPUs";
    homepage = "https://github.com/AMD-AGI/Hyperloom";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "hyperloom";
  };
}
