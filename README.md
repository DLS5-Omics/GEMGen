
# GEMGen

**GEMGen** is a generative model for **phenotype-based drug discovery**.  
It consists of two core components:

- **Generator**: generates small-molecule candidates conditioned on *cell type information* and *gene up-/down-regulation lists*.
- **Scorer**: evaluates how well a generated small molecule matches the given phenotypic input.

The framework is designed for flexible inference and can be easily adapted to different biological contexts and datasets.

---

## Installation

### 1. Create a Conda Environment

```bash
conda create -n gemgen python=3.10 -y
conda activate gemgen
```

### 2. Install Dependencies

GEMGen relies on `vllm`, whose dependencies cover all required Python packages.

```bash
pip install vllm==0.7.3
```

> **Note**
> This model only supports inference on NVIDIA GPUs. A CUDA-enabled environment with a compatible NVIDIA GPU is required.

---

## Model Checkpoints

Pretrained model checkpoints are required for both the generator and the scorer.

1. Download the checkpoint folder from **[xxxxx website]**.
2. Place the checkpoint directory at a desired local path.
3. Update the corresponding checkpoint paths in the demo scripts:

   * `demo/run_generator.sh`
   * `demo/run_scorer.sh`

---

## Data Preparation

GEMGen expects structured input data describing phenotypic perturbations.

* **Generator inputs**

  * Cell type information
  * Lists of up-regulated and down-regulated genes
  * Prompt templates (see `data/generator_prompts.txt` and `data/templates.txt`)

* **Scorer inputs**

  * Generated molecules (e.g., SMILES)
  * Corresponding phenotypic conditions
  * Example format is provided in `data/scorer_test_demo.tsv`

Please refer to the files in the `data/` directory for templates and example data formats.

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/DLS5-Omics/GEMGen.git
cd GEMGen
```

---

### 2. Run the Generator

The generator produces candidate small molecules conditioned on phenotypic inputs.

```bash
bash demo/run_generator.sh
```

Typical outputs include:

* Generated molecular SMILES

---

### 3. Run the Scorer

The scorer evaluates the consistency between generated molecules and the input phenotypic signatures.

```bash
bash demo/run_scorer.sh
```

Typical outputs include:

* Matching scores between molecules and phenotypic inputs

---

## Example Workflow

1. Prepare phenotypic input data (cell type + gene regulation).
2. Run the generator to produce candidate molecules.
3. Feed generated molecules into the scorer.
4. Select high-scoring molecules for downstream analysis or validation.

---

## License

This project is released under the terms of the license provided in the `LICENSE` file.