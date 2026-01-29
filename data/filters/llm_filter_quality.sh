#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=portuguese-quality-filter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                          # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                             # <-- Change to your filesystem
workspace_name="nanotronics"                     # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-edu-quality.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-edu-quality.$SLURM_JOB_ID"

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
export SYSTEM="Você é um avaliador de qualidade de textos para a língua portuguesa. Sua tarefa é avaliar, em uma escala de 1 a 5, o quão informativo e educativo é um dado trecho de texto. Você DEVE apresentar sua pontuação no formato JSON.

### Critérios de Avaliação:

-   Dê 1 se o texto não for informativo ou educativo. Também dê 1 se o texto for muito simples, muito curto, mal formatado, sem sentido ou contiver conteúdo NSFW.
-   Dê 2 se o texto for um pouco informativo, mas carecer de valor educacional. Por exemplo, pode misturar conteúdo educativo com material não educativo, oferecendo uma visão superficial de tópicos potencialmente úteis.
-   Dê 3 se o texto for informativo e adequado para uso educacional, apresentando conceitos-chave relevantes para os currículos escolares. Por exemplo, se o texto for um artigo bem escrito sobre um tópico científico, mas pode não ser completo ou incluir informações supérfluas, sendo excessivamente complexo ou muito simples.
-   Dê 4 se o texto for educativo e informativo, proporcionando um conteúdo altamente relevante e benéfico para fins educacionais, para um nível não superior ao ensino fundamental, exibindo um estilo de escrita claro e consistente. Por exemplo, poderia ser similar a um capítulo de livro didático ou a um tutorial, oferecendo conteúdo educacional substancial, incluindo exercícios e soluções, com informações irrelevantes mínimas.
-   Dê 5 se o texto for altamente educativo e informativo. Para uma pontuação 5, o texto deve ser excepcional em seu valor educacional, perfeitamente adequado para ensino no ensino fundamental ou ensino médio. Ele segue um raciocínio detalhado, o estilo de escrita é fácil de entender e oferece insights profundos e completos sobre o assunto.

### Requisitos de Saída:

-   Sua saída deve ser um objeto JSON válido.
-   O JSON deve conter apenas um par chave-valor.
-   A chave deve ser 'score'.
-   O valor deve ser um número inteiro entre 1 e 5.
-   O valor deve ser uma representação fiel dos critérios acima.

### Exemplo de Saída:

{'score': 2}

Certifique-se de que suas respostas sigam estritamente este formato e avaliem o texto de forma precisa, com base nos critérios fornecidos."
export PROMPT_PREFIX="Abaixo está um trecho de uma página da web. Avalie se a página tem um alto valor educacional e pode ser útil em um ambiente educacional para ensino do ensino fundamental ao ensino médio. Atribua uma pontuação entre 1 e 5. Certifique-se de que sua pontuação reflita quão informativo e educativo o texto é. O texto será fornecido em português.\n\nTexto: "
export PROMPT_SUFFIX="\n\nForneça sua pontuação como um número inteiro entre 1 e 5. Responda apenas no formato JSON."

export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
export COLUMN_NAME="text"
export DATASET_PATH="$workdir/gigaverbo-v2-dedup/classification_dataset"
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
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution (Parallel Filtering)
#############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 $workdir/llm_filter.py \
    --model_name "$MODEL_NAME" \
    --tensor_parallel_size 4 \
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
