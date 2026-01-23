#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=portuguese-toxicity-filter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                          # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                             # <-- Change to your filesystem
workspace_name="nanotronics"                     # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_parsing"
cd "$workdir"
ulimit -c 0

out="$workdir/run_parsing/out-toxicity-filter.$SLURM_JOB_ID"
err="$workdir/run_parsing/err-toxicity-filter.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"

export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export SYSTEM="Você é um filtro de toxicidade para a língua portuguesa. Sua tarefa é avaliar, em uma escala de 1 a 5, o quão tóxico (ou seja, nocivo, ofensivo ou inapropriado) é um determinado trecho de texto. Você DEVE fornecer sua pontuação em formato JSON.

### Critérios de Avaliação:

- Dê 1 se o texto não for tóxico, for neutro e apropriado para todos os públicos. O texto não contém conteúdo ofensivo, prejudicial ou inflamatório.
- Dê 2 se o texto apresentar toxicidade leve, como linguagem levemente inapropriada, leve grosseria ou sentimento negativo sutil, mas sem ser explicitamente ofensivo ou prejudicial.
- Dê 3 se o texto contiver toxicidade moderada, incluindo alguns insultos, linguagem ofensiva ou frases inadequadas, mas sem ser altamente agressivo ou gravemente nocivo.
- Dê 4 se o texto for altamente tóxico, contendo discurso de ódio explícito, ameaças diretas, insultos severos ou estereótipos prejudiciais. Pode também incluir linguagem depreciativa ou agressiva direcionada a indivíduos ou grupos.
- Dê 5 se o texto for extremamente tóxico, exibindo o mais alto nível de conteúdo nocivo, como incitação explícita à violência, discriminação grave, ameaças ou linguagem extremamente ofensiva. Essa pontuação é reservada para os textos mais perigosos e abusivos.

### Requisitos de Saída:

- Sua saída deve ser um objeto JSON válido.
- O JSON deve conter apenas um par chave-valor.
- A chave deve ser 'score'.
- O valor deve ser um número inteiro entre 1 e 5.
- O valor deve representar fielmente os critérios acima.

### Exemplo de Saída:

{'score': 2}

Certifique-se de que suas respostas sigam estritamente este formato e avaliem o texto com precisão com base nos critérios fornecidos."
export PROMPT_PREFIX="Abaixo está um trecho retirado de uma página da web. Avalie se o trecho contém linguagem nociva, ofensiva ou inapropriada. Atribua uma pontuação entre 1 e 5. Certifique-se de que sua pontuação reflita o nível de toxicidade presente no texto. O texto será fornecido em português.\n\nTexto: "
export PROMPT_SUFFIX="\n\nForneça sua pontuação como um número inteiro entre 1 e 5. Responda apenas no formato JSON."

export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
export COLUMN_NAME="text"
export DATASET_PATH="$workdir/gigaverbo-v2-dedup/toxic_classification_dataset"
export MAX_LENGTH=150
export MAX_CHUNK_SIZE=5000
export TEMPERATURE=0.2
export TOP_K=50
export TOP_P=0.9
export REPETITION_PENALTY=1.2
export NUM_RETURN_SEQUENCES=1
export ROW_START=0

hf auth login --token "$HF_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 $workdir/llm_filter.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH/chunk_0.jsonl" \
    --column_name "$COLUMN_NAME" \
    --$DATASET_PATH "$$DATASET_PATH" \
    --output_file "chunk_scored_0.jsonl" \
    --max_length $MAX_LENGTH \
    --max_chunk_size $MAX_CHUNK_SIZE \
    --chunk_once \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --repetition_penalty $REPETITION_PENALTY \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    --row_start $ROW_START 1>$out 2>$err 

#############################################
# End of Script
#############################################
# Clean cache folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"