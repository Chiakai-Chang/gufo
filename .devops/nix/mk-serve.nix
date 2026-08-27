{
  lib,
  writeShellApplication,
  strix,
}:

{
  name ? "strix-serve-${modality}",
  modality ? "llm", # "llm", "video", or "audio"
  model, # path, derivation, or string to GGUF model or weights directory

  # Server Options
  host ? "127.0.0.1",
  port ? 8080,
  sessions ? null,
  maxConnections ? null,
  maxRequestBytes ? null,
  apiKey ? null,
  verbose ? false,

  # LLM Options
  servedModelName ? null,
  context ? null,
  maxTokens ? null,
  temperature ? null,
  topP ? null,
  topK ? null,
  minP ? null,
  seed ? null,
  repeatPenalty ? null,
  repeatLastN ? null,
  system ? null,
  raw ? false,
  chatTemplate ? null,
  think ? null, # "on", "off", "auto"
  reasoningBudget ? null,
  speculative ? null, # "mtp", "mtp-npu", "npu", "pld", "self"
  draftModel ? null, # mtp-model path / derivation
  mtpModel ? draftModel,
  draftTokens ? null,
  prefillChunk ? null,
  maxPending ? null,
  maxPendingPerClient ? null,
  requestTimeoutMs ? null,
  maxOutputBytes ? null,
  maxBufferedOutputBytes ? null,
  maxBufferedOutputTotal ? null,
  cpu ? false,

  # Video Options
  root ? null,
  manifest ? null,
  ttl ? null,

  # Custom strix package override
  strixPackage ? strix,
  extraArgs ? [ ],
}:

let
  validModalities = [
    "llm"
    "video"
    "audio"
  ];
  validThinkModes = [
    "on"
    "off"
    "auto"
  ];
  validSpeculativeModes = [
    "mtp"
    "mtp-npu"
    "npu"
    "pld"
    "self"
  ];

  serverArgs =
    [
      "--host"
      host
      "--port"
      (toString port)
    ]
    ++ lib.optionals (sessions != null) [
      "--sessions"
      (toString sessions)
    ]
    ++ lib.optionals (maxConnections != null) [
      "--max-connections"
      (toString maxConnections)
    ]
    ++ lib.optionals (maxRequestBytes != null) [
      "--max-request-bytes"
      (toString maxRequestBytes)
    ]
    ++ lib.optionals (apiKey != null) [
      "--api-key"
      apiKey
    ]
    ++ lib.optionals verbose [ "--verbose" ];

  modalityArgs =
    [
      "--model"
      (toString model)
    ]
    ++ lib.optionals (modality == "llm") (
      lib.optionals (servedModelName != null) [
        "--served-model-name"
        servedModelName
      ]
      ++ lib.optionals (context != null) [
        "--context"
        (toString context)
      ]
      ++ lib.optionals (maxTokens != null) [
        "--max-tokens"
        (toString maxTokens)
      ]
      ++ lib.optionals (temperature != null) [
        "--temperature"
        (toString temperature)
      ]
      ++ lib.optionals (topP != null) [
        "--top-p"
        (toString topP)
      ]
      ++ lib.optionals (topK != null) [
        "--top-k"
        (toString topK)
      ]
      ++ lib.optionals (minP != null) [
        "--min-p"
        (toString minP)
      ]
      ++ lib.optionals (seed != null) [
        "--seed"
        (toString seed)
      ]
      ++ lib.optionals (repeatPenalty != null) [
        "--repeat-penalty"
        (toString repeatPenalty)
      ]
      ++ lib.optionals (repeatLastN != null) [
        "--repeat-last-n"
        (toString repeatLastN)
      ]
      ++ lib.optionals (system != null) [
        "--system"
        system
      ]
      ++ lib.optionals raw [ "--raw" ]
      ++ lib.optionals (chatTemplate != null) [
        "--chat-template"
        chatTemplate
      ]
      ++ lib.optionals (think != null) [
        "--think"
        think
      ]
      ++ lib.optionals (reasoningBudget != null) [
        "--reasoning-budget"
        (toString reasoningBudget)
      ]
      ++ lib.optionals (speculative != null) [
        "--speculative"
        speculative
      ]
      ++ lib.optionals (mtpModel != null) [
        "--mtp-model"
        (toString mtpModel)
      ]
      ++ lib.optionals (draftTokens != null) [
        "--draft-tokens"
        (toString draftTokens)
      ]
      ++ lib.optionals (prefillChunk != null) [
        "--prefill-chunk"
        (toString prefillChunk)
      ]
      ++ lib.optionals (maxPending != null) [
        "--max-pending"
        (toString maxPending)
      ]
      ++ lib.optionals (maxPendingPerClient != null) [
        "--max-pending-per-client"
        (toString maxPendingPerClient)
      ]
      ++ lib.optionals (requestTimeoutMs != null) [
        "--request-timeout-ms"
        (toString requestTimeoutMs)
      ]
      ++ lib.optionals (maxOutputBytes != null) [
        "--max-output-bytes"
        (toString maxOutputBytes)
      ]
      ++ lib.optionals (maxBufferedOutputBytes != null) [
        "--max-buffered-output-bytes"
        (toString maxBufferedOutputBytes)
      ]
      ++ lib.optionals (maxBufferedOutputTotal != null) [
        "--max-buffered-output-total"
        (toString maxBufferedOutputTotal)
      ]
      ++ lib.optionals cpu [ "--cpu" ]
    )
    ++ lib.optionals (modality == "video") (
      lib.optionals (root != null) [
        "--root"
        (toString root)
      ]
      ++ lib.optionals (manifest != null) [
        "--manifest"
        (toString manifest)
      ]
      ++ lib.optionals (ttl != null) [
        "--ttl"
        (toString ttl)
      ]
    )
    ++ lib.optionals (modality == "audio") (
      lib.optionals (context != null) [
        "--context"
        (toString context)
      ]
    )
    ++ extraArgs;
in
assert lib.assertMsg (lib.elem modality validModalities)
  "strix.mkServe: 'modality' must be one of ${lib.generators.toJSON { } validModalities}, got '${modality}'";
assert lib.assertMsg (model != null && model != "")
  "strix.mkServe: 'model' must be specified (cannot be empty)";
assert lib.assertMsg (think == null || lib.elem think validThinkModes)
  "strix.mkServe: 'think' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString think}'";
assert lib.assertMsg (speculative == null || lib.elem speculative validSpeculativeModes)
  "strix.mkServe: 'speculative' must be one of ${lib.generators.toJSON { } validSpeculativeModes}, got '${toString speculative}'";
writeShellApplication {
  inherit name;
  runtimeInputs = [ strixPackage ];
  text = ''
    exec ${strixPackage}/bin/strix serve ${lib.escapeShellArgs serverArgs} ${modality} ${lib.escapeShellArgs modalityArgs} "$@"
  '';
}
