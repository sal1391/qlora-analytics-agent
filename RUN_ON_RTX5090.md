# Run full Qwen2.5-3B QLoRA on your Acer Helios (RTX 5090)

Your machine (32 GB VRAM) is the primary target. This is free locally and well under a $100 cloud budget.

## 1. Setup (once)

```powershell
cd qlora-analytics-agent
conda env create -f environment.yml
conda activate qlora-analytics-agent
# or: pip install -r requirements.txt
```

## 2. Data (already generated; regenerate if needed)

```bash
python scripts/01_generate_synthetic_db.py
python scripts/02_generate_instruction_data.py
python scripts/03_preprocess.py
```

## 3. Train QLoRA (main paper experiment)

```bash
# Main config r=16
python scripts/04_train_qlora.py --config configs/train_qlora_3b.yaml --lora_r 16

# Ablations
python scripts/04_train_qlora.py --lora_r 8
python scripts/04_train_qlora.py --lora_r 32
```

Expected wall time on RTX 5090: roughly 30–90 minutes depending on epochs/settings.

## 4. Evaluate + figures

```bash
python scripts/05_evaluate.py
python scripts/06_plot_results.py
```

Then rebuild the PDF:

```bash
cd latex
pdflatex main.tex
pdflatex main.tex
```

## 5. Connect work Snowflake later (private)

```bash
export SNOWFLAKE_ACCOUNT=...
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_WAREHOUSE=...
export SNOWFLAKE_DATABASE=...
export SNOWFLAKE_SCHEMA=...
export SNOWFLAKE_ROLE=...
```

See `capture/DATA_CAPTURE_GUIDE.md` for fields to log and privacy rules.

## Colab fallback

Open `notebooks/Colab_QLoRA_Training.ipynb`. A100/T4 runs should stay far under $100.
