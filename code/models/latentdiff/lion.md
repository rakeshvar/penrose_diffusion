
# LION
```mermaid
flowchart LR

%% Inputs
subgraph INPUT1["Input"]
XYA@{ shape: lin-rect, label: "XYA" }
COLORS@{ shape: lin-rect, label: "Colors" }
CLASS@{ shape: lin-rect, label: "Class" }
end

XYA2@{ shape: lin-rect, label: "XYA
Colors" }
COLORS2@{ shape: lin-rect, label: "Colors" }
CLASS2@{ shape: lin-rect, label: "Class
drop" }

%% Encoder
subgraph ENC["Set Encoder"]
  QE(("Q"))
  TWMLP["Tile-wise
    MLP"]
  ATTNPOOL["Attention
    Pooling"]
  CLSEMBD["Class
    Embedding"]
  FCL["FCL"]
end

COLORS --> TWMLP
QE --> ATTNPOOL
XYA --> TWMLP --> ATTNPOOL --> FCL
CLASS --> CLSEMBD --> FCL
FCL --> MUSIGMA2(["μ,σ²"]) --> LKL{{"KL Loss"}}

subgraph REPARAMETRIZE["Reparametrize"]
  REPARAM(["μ + σ ⊙ ε"])
end
MUSIGMA2 --> REPARAM --> Z0(("Z0"))

subgraph DIFFUSION["Latent Diffusion"]
  EPS(("ε"))
  FWDIFF(["Forward
    Diffusion"])
  DENOISER["Denoiser"]
  ZT(("Zₜ"))
  T(("t"))
end

EPS --> FWDIFF
EPS --> LDIFF
T --> FWDIFF
T --> DENOISER

CLASS2 --> DENOISER
Z0 --> FWDIFF --> ZT --> DENOISER --> EPSHAT([" ̂ϵ"])  --> LDIFF

%% Decoder
subgraph DEC["Decoder"]
  DECODER["Decoder"]
  Q
end
Z0 --> DECODER
Q(("Q")) --> DECODER

DECODER --> Tiles(["xya^"]) --> LREC
COLORS2 --> DECODER

%% Losses
LREC{{"Reconstruction
Loss"}}
LDIFF{{"Diffusion
Loss"}}
XYA2 --> LREC
subgraph LOSSES["Losses"]
 LREC
 LKL
 LDIFF
end
```

## Sampling
```mermaid
flowchart LR

%% Inputs
subgraph Input
  COLORS@{ shape: lin-rect, label: "Colors" }
  CLASS@{ shape: lin-rect, label: "Class" }
end

NOISE(["zₜ  ~ 𝒩 (0, I)"]) --> Denoiser
CLASS --> Denoiser
Denoiser --> Z0(("z₍ₜ₋₁₎")) --> Decoder
Z0 --> Denoiser
COLORS --> Decoder --> X(("Sample"))

```

## FILM

```mermaid
graph TB
    subgraph "Traditional Approach"
        A1[Input x] --> Add1["+"]
        A2[Conditioning] --> Add1
        Add1 --> A3[Network]
        A3 --> A4[Output]
    end
```

```mermaid
graph TB
    subgraph "FiLM Block"
        Input[/"zₜ"/]
        Cond[/"Class, Time"/]

        Input --> Linear["Linear "]
        Linear --> Norm["LayerNorm"]

        Cond --> FiLMNet["Linear"]
        FiLMNet --> Split["Split"]
        Split --> Scale["Scale γ"]
        Split --> Shift["Shift β"]

        Norm --> Modulate["β + (1+γ)h"]
        Scale --> Modulate
        Shift --> Modulate

        Modulate --> Act["Activation"]
        Act --> Output[/"Output"/]
    end
```