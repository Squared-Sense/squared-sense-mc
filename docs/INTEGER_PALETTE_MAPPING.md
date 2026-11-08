# Integer Palette Mapping (Task 9)

## 1. Executive Summary & Design Rationale

In reinforcement learning pipelines, observational block identity is consumed by a neural network via an embedding table (`torch.nn.Embedding`). Passing strings (`"minecraft:stone"`) or unstandardized integer IDs introduces major synchronization and performance issues:

1. **Strings are unusable on GPUs**: Every string would need repeated hashing or dictionary lookup on the Python side before creating tensors, adding substantial CPU latency to the RL training loop.
2. **Minecraft runtime `getRawId()` is non-deterministic**: The internal integer IDs assigned by the Minecraft engine can shift between minor game versions, modded environments, or world reload orders. Relying on dynamic internal IDs would desynchronize the Java client from the PyTorch model weights.
3. **Static Canonical Freezing**: Squared Sense MC builds a **frozen, deterministic dictionary** of all Minecraft blocks at startup. Blocks are sorted strictly alphabetically by their canonical registry name (`minecraft:acacia_boat`, `minecraft:stone`, etc.) and assigned sequential integers.
4. **Shared `block_vocab.json`**: The dictionary is exported directly to disk alongside observations, ensuring that PyTorch embedding tables are guaranteed to be 1:1 synchronized with the mod's output.

---

## 2. Token ID Assignment Specification

The vocabulary reserves specific low indices for fundamental states and assigns sequential tokens to all registered blocks:

| Token ID | Identifier | Description | Semantic Purpose |
|---|---|---|---|
| **`0`** | `minecraft:air` | Empty space / Air | Collapses `minecraft:air`, `minecraft:cave_air`, and `minecraft:void_air`. Natural padding index for embeddings. |
| **`1`** | `special:masked` | Unseen / Occluded | Reserved for Phase 5 visibility culling (underworld shell filter). Represents unseen voxels hidden behind solid terrain. |
| **`2`** | `minecraft:acacia_button` | First alphabetical block | Sorted alphabetical registry names start at index 2. |
| **`3`** | `minecraft:acacia_chest_boat` | Second alphabetical block | |
| **...** | ... | ... | ... |
| **`N`** | `minecraft:zombified_piglin_spawn_egg` | Last alphabetical block | Total vocabulary size is $N + 1 \approx 880\text{--}900$ tokens in vanilla 1.21.1. |

### Vocabulary Size & Memory Footprint

In Minecraft 1.21.1 vanilla:
- **Non-air blocks registered**: ~880
- **Reserved tokens**: 2 (`0 = air`, `1 = masked`)
- **Total `vocabSize`**: ~882 tokens

Because 882 fits comfortably within an unsigned 16-bit integer (`uint16`, max 65,535), voxel tokens require only **2 bytes per cell** in binary tensor representations.

In PyTorch, an embedding table mapping 882 tokens to an 8-dimensional or 16-dimensional continuous vector:
$$\text{VRAM} = 882 \times 8 \times 4\text{ bytes} \approx 28.2\text{ KB}$$
This lookup is an $O(1)$ array indexing operation on the GPU with virtually zero memory or computational footprint.

---

## 3. Java Mod Architecture (`org.sqsense...observation.palette`)

The palette mapping is implemented in [`BlockVocabulary.java`](file:///c:/Users/Krishna/Projects/Squared%20Sence%20MC/src/client/java/org/sqsense/squared_sense_mc/client/observation/palette/BlockVocabulary.java).

```
src/client/java/org/sqsense/squared_sense_mc/client/observation/
└── palette/
    └── BlockVocabulary.java    # Canonical vocabulary dictionary, O(1) lookups & JSON exporter
```

### High-Speed $O(1)$ Lookup Structures
At runtime during 20-tick observation sampling, resolving a `Block` instance to its integer token ID must not incur string concatenation or map overhead:

1. **`IdentityHashMap<Block, Integer> BLOCK_TO_ID`**:
   - Because `Block` instances are immutable singletons in Minecraft (`Blocks.STONE`, `Blocks.DIRT`), pointer identity equality (`==`) is used instead of `.equals()`.
   - Yields instantaneous $O(1)$ resolution with zero GC allocations.
2. **Flat Arrays for Reverse Lookup**:
   - `String[] ID_TO_NAME`: Resolves `id -> "minecraft:stone"` via array index.
   - `Block[] ID_TO_BLOCK`: Resolves `id -> Blocks.STONE` via array index.
3. **`Map<String, Integer> NAME_TO_ID`**:
   - Fallback map for string-based queries.

### Air Collapsing
Any variant of air (`minecraft:air`, `minecraft:cave_air`, `minecraft:void_air`) is mapped directly to `AIR_ID = 0`:
```java
NAME_TO_ID.put("minecraft:air", 0);
NAME_TO_ID.put("minecraft:cave_air", 0);
NAME_TO_ID.put("minecraft:void_air", 0);
BLOCK_TO_ID.put(Blocks.AIR, 0);
BLOCK_TO_ID.put(Blocks.CAVE_AIR, 0);
BLOCK_TO_ID.put(Blocks.VOID_AIR, 0);
```

---

## 4. Export Format & Commands

### Automatic Export
Whenever an observation is serialized to the `observations/` directory, [`ObservationJsonSerializer.java`](file:///c:/Users/Krishna/Projects/Squared%20Sence%20MC/src/client/java/org/sqsense/squared_sense_mc/client/observation/serialization/ObservationJsonSerializer.java) automatically checks if `observations/block_vocab.json` exists. If not present, it exports the canonical vocabulary immediately.

### In-Game Command: `/observe vocab`
Players or automated scripts can explicitly regenerate the vocabulary JSON at any time:
```
/observe vocab
/obs vocab
```
Feedback provides a clickable link to open `block_vocab.json`.

### `block_vocab.json` Schema
The exported JSON contains both forward (`idToBlock`) and reverse (`blockToId`) mappings:

```json
{
  "version": 1,
  "vocabSize": 882,
  "reservedTokens": {
    "0": "minecraft:air",
    "1": "special:masked"
  },
  "idToBlock": {
    "0": "minecraft:air",
    "1": "special:masked",
    "2": "minecraft:acacia_button",
    "3": "minecraft:acacia_chest_boat",
    "782": "minecraft:stone"
  },
  "blockToId": {
    "minecraft:acacia_button": 2,
    "minecraft:acacia_chest_boat": 3,
    "minecraft:air": 0,
    "minecraft:cave_air": 0,
    "minecraft:void_air": 0,
    "special:masked": 1,
    "minecraft:stone": 782
  }
}
```

---

## 5. Observation JSON Schema Update

Each voxel cell in the observation grid now includes the integer token `id` alongside `name` and physical channels:

```json
{
  "metadata": {
    "gridSize": 16,
    "vocabSize": 882,
    "channels": [
      "id",
      "name",
      "dx",
      "dy",
      "dz",
      "occupancy",
      "passability",
      "hazard",
      "hardness",
      "lightLevel",
      "fluidFlow"
    ]
  },
  "grid": [
    {
      "id": 782,
      "name": "minecraft:stone",
      "dx": -2,
      "dy": -1,
      "dz": 4,
      "properties": {
        "occupancy": 1.0,
        "passability": 0.0,
        "hazard": 0.0,
        "hardness": 1.5,
        "lightLevel": 0.0,
        "fluidFlow": [0.0, 0.0]
      }
    }
  ]
}
```

---

## 6. PyTorch / RL Integration Example

Here is how downstream Python scripts load `block_vocab.json` and feed block integer IDs into a PyTorch model:

```python
import json
import torch
import torch.nn as nn

# 1. Load canonical vocabulary
with open("observations/block_vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

vocab_size = vocab_data["vocabSize"]
print(f"Loaded block vocabulary: {vocab_size} tokens")

# 2. Define 3D Voxel Perception Model
class VoxelEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 8):
        super().__init__()
        # Token ID embedding layer (padding_idx=0 treats air as zero vector)
        self.block_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        
        # 7 continuous physical channels:
        # occupancy(1) + passability(1) + hazard(1) + hardness(1) + light(1) + fluid_flow(2) = 7
        continuous_channels = 7
        total_channels = embed_dim + continuous_channels
        
        # 3D Convolutional backbone
        self.conv3d = nn.Sequential(
            nn.Conv3d(total_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # -> 8x8x8
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, block_ids: torch.Tensor, physical_channels: torch.Tensor) -> torch.Tensor:
        """
        block_ids: LongTensor of shape (B, 16, 16, 16)
        physical_channels: FloatTensor of shape (B, 7, 16, 16, 16)
        """
        # Embed integer IDs: (B, 16, 16, 16) -> (B, 16, 16, 16, embed_dim)
        embedded = self.block_embedding(block_ids)
        # Permute to channels-first: (B, embed_dim, 16, 16, 16)
        embedded = embedded.permute(0, 4, 1, 2, 3)
        
        # Concatenate embedding with continuous physical channels
        x = torch.cat([embedded, physical_channels], dim=1) # (B, 15, 16, 16, 16)
        return self.conv3d(x)

# Example usage:
model = VoxelEncoder(vocab_size=vocab_size, embed_dim=8)
sample_ids = torch.randint(0, vocab_size, (1, 16, 16, 16), dtype=torch.long)
sample_physical = torch.zeros((1, 7, 16, 16, 16), dtype=torch.float32)

feature_vector = model(sample_ids, sample_physical)
print(f"Output shape: {feature_vector.shape}")
```

---

## 7. Semantic Bucketing vs. Full Vocabulary

In reinforcement learning, having ~900 tokens has zero negative computational impact because an embedding lookup is $O(1)$. However, rare decorative blocks (e.g. `pink_glazed_terracotta`, `magenta_stained_glass`) may appear infrequently in training trajectories.

If faster early training convergence is desired, downstream Python preprocessing can collapse cosmetic variants to base semantic IDs **without requiring changes to the Java mod**:

| Category | Vanilla Blocks | Semantic Collapsed Base |
|---|---|---|
| **Wool** | 16 dyed colors (`white_wool`, `red_wool`...) | `minecraft:white_wool` |
| **Concrete** | 16 dyed colors | `minecraft:white_concrete` |
| **Terracotta** | 16 dyed colors | `minecraft:terracotta` |
| **Stone Variants** | Granite, Diorite, Andesite, Tuff | `minecraft:stone` |
| **Wood Slabs/Stairs** | Oak, Birch, Spruce, Dark Oak... | `minecraft:oak_slab`, `minecraft:oak_stairs` |

For v1, using the full ~882 canonical token IDs is recommended as it preserves complete environmental fidelity and lets backpropagation learn embeddings naturally.

