#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=portuguese-instruct-quality-filter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --dependency=afterany:22394479      # <-- Change dependency as needed

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

out="$workdir/run_outputs/out-instruct-quality.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-instruct-quality.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh
source $workdir/.venv_amd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export SYSTEM="Você é um avaliador de respostas de assistente. Sua tarefa é avaliar, em uma escala de 1 a 5, o quão bem o assistente seguiu a instrução do usuário em uma determinada interação. Você DEVE fornecer sua pontuação em formato JSON.

Cada amostra de entrada consiste em uma consulta do usuário (com ou sem um prompt de sistema) e a resposta do assistente. Todas essas conversas estarão principalmente em português (exceto em casos de tarefas de tradução). Chamadas de ferramentas podem estar envolvidas na conversa. Sua avaliação deve se concentrar em quão eficaz e fielmente a resposta do assistente atende à solicitação do usuário.

### Critérios de Avaliação:

- Dê 1 se a resposta do assistente for completamente irrelevante, ignorar a instrução do usuário, contiver conteúdo prejudicial, sem sentido ou impróprio (NSFW), não responder de maneira coerente ou mudar de idioma no meio da geração sem motivo ou instrução. A pontuação 1 também é apropriada se a solicitação do usuário for malformada, sem sentido ou impossível de cumprir.
- Dê 2 se o assistente tentou seguir a instrução, mas entendeu mal a intenção principal ou forneceu uma resposta incompleta ou pouco útil.
- Dê 3 se o assistente cumpriu parcialmente a instrução, mas com alguns erros, elementos ausentes ou execução imprecisa. A ideia central está presente, mas a resposta carece de clareza, profundidade ou alinhamento completo com a solicitação.
- Dê 4 se o assistente seguiu bem a instrução, com apenas pequenas omissões ou imprecisões. A resposta está quase completa e é útil, alinhando-se com a intenção e expectativa do usuário.
- Dê 5 se o assistente seguiu total e precisamente a instrução do usuário. A resposta está completa, precisa e formatada adequadamente, demonstrando excelente compreensão da tarefa.

### Requisitos de Fundamentação da Avaliação:

- Se a resposta do assistente contiver tags de raciocínio vazias (<think></think>), você deve ignorá-las e focar no conteúdo fora dessas tags.
- Se a resposta do assistente contiver raciocínio dentro das tags <think></think>, você deve avaliar o conteúdo da resposta após essas tags.
- Se a resposta do assistente contiver chamadas de ferramentas, você deve avaliar se a chamada da ferramenta foi apropriada e se a resposta do assistente após a chamada é coerente e relevante para a solicitação do usuário.
- Se a resposta do assistente fornecer esclarecimentos (por exemplo, que as ferramentas atuais não são adequadas para a tarefa), você deve considerar se a resposta do assistente é transparente quanto às limitações.
- Se a resposta do assistente estiver em um idioma diferente da solicitação do usuário, você deve avaliar se a mudança de idioma foi apropriada e alinhada com a intenção do usuário.

### Requisitos de Saída:

- Sua saída deve ser um objeto JSON válido.
- O JSON deve conter dois pares chave-valor.
- Um deve se chamar 'score' e o outro 'reason'.
- O 'score' deve ser um número inteiro entre 1 e 5.
- O 'reason' deve ser uma explicação breve da sua pontuação, resumindo o quão bem a resposta do assistente atendeu à solicitação do usuário.
- O valor deve representar fielmente os critérios acima.

### Exemplo de Saída:

{'score': 2, 'reason': 'O assistente forneceu uma resposta parcial, mas não abordou completamente a solicitação do usuário por uma explicação detalhada.'}

Certifique-se de que sua avaliação siga estritamente este formato e reflita com precisão o quão bem a resposta do assistente atendeu à solicitação do usuário."

export PROMPT_PREFIX="Abaixo está uma conversa entre um usuário e um assistente. Avalie o quão bem o assistente seguiu as instruções do usuário. Atribua uma pontuação entre 1 e 5 de acordo com os critérios de avaliação estabelecidos anteriormente. Certifique-se de que sua pontuação reflita o quão bem a resposta do assistente atende à solicitação do usuário. O texto será fornecido sempre em português.\n\n---\n\n"
export PROMPT_SUFFIX="\n---\n\nForneça sua avaliação. Responda SOMENTE em formato JSON."

export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
export COLUMN_NAME="text"
export OUTPUT_DIR="$workdir/portuguese/portuguese-instruct-qwen-annotations"
export MAX_LENGTH=200
export MAX_CHUNK_SIZE=7000
export TEMPERATURE=0.2
export TOP_K=50
export TOP_P=0.9
export REPETITION_PENALTY=1.2
export NUM_RETURN_SEQUENCES=1

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
python3 "$workdir/llm_filter.py" \
    --model_name "$MODEL_NAME" \
    --tensor_parallel_size 4 \
    --dataset_path "$workdir/portuguese/portuguese-instruct-qwen-annotations/chunk_3.jsonl" \
    --column_name "$COLUMN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "chunk_scored_3.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>>"$out" 2>>"$err"

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