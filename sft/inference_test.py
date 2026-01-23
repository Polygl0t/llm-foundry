"""
Inference Testing and Evaluation Suite for Language Models

This script performs comprehensive inference testing on trained language models across multiple task types:
- Classification (sentiment analysis, category detection)
- Structured Output (JSON generation and validation)
- Function Call / Tool Use (tool calling capabilities)
- Summarization (content condensation)
- Reasoning (step-by-step thinking with <think> tags)
- Math reasoning, Translation, Retrieval, and more

Example usage:
    python inference_test.py \
        --model_path checkpoints/llama-sft/final \
        --samples_file custom_samples.json \
        --output_file results/evaluation.json \
        --max_new_tokens 1024 \
        --temperature 0.1
"""
import json
import re
import torch
import argparse
import json_repair
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

DEFAULT_SAMPLES = [
    {
        "messages": [
            {
                "role": "system",
                "content": "Você é um mentor técnico com foco em programação. Ensine o usuário com exemplos, analogias e linguagem acessível."
            },
            {
                "role": "user",
                "content": "Como eu posso abrir um arquivo CSV em Python usando a biblioteca pandas?"
            }
        ],
        "task_type": "Coding",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Leia o conteúdo e produza um resumo em português que destaque os elementos centrais.\n\nO Tribunal Eleitoral da Bolívia retomou nesta segunda-feira (21) um sistema de contagem rápida de votos, após reclamações de opositores, da OEA e vários países, e situou o presidente Evo Morales na liderança (46,4%), seguido do opositor Carlos Mesa (37,07%), com 95,09% das cédulas apuradas.  Entre Morales e Mesa, segundo um informe público do sistema TREP (Transmissão de Resultados Eleitorais Preliminares), controlado pelo Tribunal Supremo Eleitoral (TSE), há uma diferença de 9,33 pontos e o chefe de Estado estaria a 0,67 ponto de vencer no primeiro turno e, desta forma, evitar um segundo turno que a oposição reivindica.  Segundo a lei boliviana, um candidato vence no primeiro turno se obtiver 50% mais um dos votos ou superar os 40% com dez pontos de diferença do segundo colocado.  Na noite de domingo, um primeiro boletim da contagem rápida, com 84% dos votos apurados pelo TREP, dava 45,28% a Morales e 38,16% a Mesa, mas o escrutínio foi paralisado até a tarde desta segunda-feira, provocando protestos de Mesa e dos observadores da Organização de Estados Americanos. Além disso, países como Brasil, Argentina e Estados Unidos pediram a reativação do TREP.  Mesa disse mais cedo nesta segunda que os resultados do TREP garantiriam um segundo turno contra Morales em dezembro, e denunciou que a situação, em cumplicidade com o TSE, está tentando manipular os votos. Por este motivo, convocou militantes e a população a se mobilizar para que seja respeitada a vontade popular.  Em La Paz, nos arredores de um luxuoso hotel onde o TSE faz a apuração dos votos, militantes do partido de Mesa, o Comunidade Cidadã, estiveram desde o meio-dia agitando bandeiras do partido e gritando para pedir respeito à votação que, insistem, assegura um segundo turno.  \"Eu vim pedir o respeito ao meu voto, é evidente que aqui houve uma fraude\", disse à AFP o jovem Alexis Romero.  Do lado oposto, Milka, uma militante pró-Morales, que não quis dar seu sobrenome, disse: \"estamos aqui para defender o voto da cidadania\".  - Potosí, estopim de protestos -  Uma missão de observadores da OEA \"rechaçou\" a interrupção da apuração de votos na região de Potosí, no sudoeste da Bolívia, onde foram registrados protestos contra o órgão eleitoral diante dos temores de fraude em benefício de Morales.  Os protestos obrigaram o Tribunal Departamental a suspender a apuração de votos na cidade, e veículos de comunicação locais revelaram que policiais se deslocavam da vizinha Sucre a Potosí, sem que o governo tenha confirmado ou informado as razões deste deslocamento.  \"A Missão de Observação Eleitoral da #OEAnaBolivia rejeita a interrupção da contagem definitiva no Tribunal Eleitoral Departamental (TED) de Potosí e faz um apelo a que se retome para dar certeza à cidadania. Urgimos que estes processos se realizem de forma pacífica\", destacou a organização no Twitter.  O tuíte foi publicado horas depois de o Ministério das Relações Exteriores informar, pela mesma rede social, que o chanceler Diego Pary e os observadores da OEA \"acordaram estabelecer uma equipe de acompanhamento permanente no processo de contagem oficial de votos\".  A missão da OEA celebrou uma reunião nesta segunda \"com o Tribunal Supremo Eleitoral (TSE) e transmitiu-lhe a necessidade de manter informada a cidadania sobre os passos seguintes na entrega dos resultados\".  Os protestos em Potosi diante da sede do tribunal eleitoral se estenderam para La Paz (oeste) e Santa Cruz (leste)."
            }
        ],
        "task_type": "Summarization",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Você é um assistente matemático treinado para resolver problemas complexos com clareza. Resolva os problemas passo a passo, explicando cada raciocínio."
            },
            {
                "role": "user",
                "content": "Como eu posso resolver o seguinte problema: 2x + 3 = 11?"
            }
        ],
        "task_type": "Math (Reasoning)",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Seu papel é traduzir conteúdos do português para o inglês. Evite traduções literais quando puder usar expressões naturais em inglês."
            },
            {
                "role": "user",
                "content": "A experiência anterior dos pacientes em DP com a outra modalidade de diálise pode ser devido ao manejo de situações de urgência."
            }
        ],
        "task_type": "Translation",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Você é um assistente que utiliza apenas dados extraídos do contexto para responder perguntas. Quando o contexto for insuficiente, diga claramente que não é possível gerar uma resposta com os dados disponíveis."
            },
            {
                "role": "user",
                "content": "Como o autor do texto se sente sobre a opinião do deputado Manuel dos Santos?\n\n<context>\nNão aceito que o Sr. Deputado Manuel dos Santos ou qualquer outro Deputado desta Câmara me julgue menos sério do que vós. Não aceito isso de ninguém e foi isso o que o Sr. Deputado Manuel dos Santos aqui disse, ao proferir aquele rol de inverdades e de excessos acerca da maioria.\n</context>"
            }
        ],
        "task_type": "Retrieval",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Você é um especialista em análise e resumo textual. Resuma os textos conforme solicitado, utilizando sempre a língua portuguesa e apresentando o resultado em formato JSON."
            },
            {
                "role": "user",
                "content": "Extraia e resuma as informações principais do e-mail abaixo. Escreva em português e organize a resposta em um JSON contendo: 'assunto', 'remetente', 'destinatário', 'resumo'. Deixe como 'null' se não encontrar alguma chave.\n\nAssunto: RE: Atualização sobre o Projeto de Conservação de Bacias Hidrográfica\n\nCarlos,\n\nObrigado por enviar o currículo atualizado. Tive a oportunidade de revisá-lo e achei que ficou ótimo! Você fez um excelente trabalho ao tornar as informações acessíveis e envolventes para uma ampla audiência.\n\nEm relação à possível resistência dos membros da comunidade, concordo que devemos estar preparados para abordar suas preocupações. Uma ideia que tive foi incluir alguns exemplos práticos de como o uso indevido das águas no rio Amazonas ou na Baía de Guanabara e o descarte incorreto de resíduos sólidos têm prejudicado essas áreas. Ouvi-los falar diretamente sobre as consequências pode ser muito mais persuasivo do que simplesmente mostrar dados estatísticos.\n\nAlém disso, também considero importante destacarmos os ganhos econômicos trazidos pela adoção de práticas mais sustentáveis, como tarifas de água menores e gastos diminutos relacionados à limpeza urbana. No final das contas, as pessoas costumam responder bem quando veem vantagens tangíveis além dos aspectos puramente ambientais.\n\nDiga-me o que você pensa dessas sugestões! Ficarei contente em ajudar a desenvolver ainda mais estratégias para vencer essa resistência e engajar as pessoas nessa nossa jornada.\n\nAtenciosamente,\nHelena"
            }
        ],
        "task_type": "Structured Output",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Você deve explicar suas decisões passo a passo. Sua resposta final só deve vir depois que o processo completo for descrito."
            },
            {
                "role": "user",
                "content": "Você pode me dar uma receita simples de bolo de chocolate?"
            }
        ],
        "task_type": "Reasoning",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Você é um assistente que resolve e revisa problemas matemáticos. Mostre o caminho completo até a resposta, incluindo conferência de unidades e coerência."
            },
            {
                "role": "user",
                "content": "Se Lucas tem 3/4 de um bolo e comeu 1/2 dele, quanto resta do bolo?"
            }
        ],
        "task_type": "Math",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Remova todas as informações irrelevantes da frase a seguir.\n\n\"O novo restaurante que abriu o centro da cidade, de propriedade do primo de John, que costumava ser chef de um restaurante com classificação Michelin em Paris, serve uma variedade de cozinhas de todo o mundo.\""
            }
        ],
        "task_type": "Rewriting",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "Aja como um assistente atencioso e explicativo. Utilize as ferramentas para resolver tarefas. Se as ferramentas forem insuficientes, explique ao usuário que não é possível completar a tarefa."
            },
            {
                "role": "user",
                "content": "Eu preciso criar uma nova tarefa para o meu projeto."
            },
            {
                "role": "assistant",
                "content": "Claro, eu posso ajudar com isso. Você poderia, por favor, me fornecer a descrição da tarefa e a data de entrega?"
            },
            {
                "role": "user",
                "content": "A tarefa é finalizar o relatório do projeto e a data de entrega é 2022-03-15."
            }
        ],
        "task_type": "Function Call / Tool Use",
        "tool" : [
            {
                "type": "function",
                "function": {
                    "name": "create_todo",
                    "description": "Cria uma nova tarefa no aplicativo de todo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Descrição da tarefa a ser criada."
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Data de entrega da tarefa no formato YYYY-MM-DD."
                            }
                        },
                        "required": ["task_description", "due_date"]
                    }
                }
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "você é uma assistente de bate-papo IA projetada para sempre falar como um pirata do caribe."
            },
            {
                "role": "user",
                "content": "Pode fornecer informações sobre que tipo de dieta um Golden Retriever deve ter?"
            }
        ],
        "task_type": "System Prompts / Steering",
        "tool" : []
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Frase 1: O artista pintou um quadro inspirado nos campos floridos da primavera.\nFrase 2: O criador produziu uma obra que representava paisagens urbanas cinzentas.\nPergunta: Qu\\u00e3o similares s\\u00e3o as duas frases? D\\u00ea uma pontua\\u00e7\\u00e3o entre 1,0 a 5,0.\nResposta:"
            }
        ],
        "task_type": "Similarity Scoring",
        "tool" : []
    },
    {   "messages": [
            {
                "role": "user",
                "content": "Classifique o sentimento da seguinte resenha de filme como 'Positiva', 'Negativa' ou 'Neutra':\n\n\"O filme foi uma experiência cinematográfica incrível, com atuações excepcionais e uma trama envolvente que me manteve na ponta da cadeira do começo ao fim.\""
            }
        ],  
        "task_type": "Classification",
        "tool" : []
    }
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a language model with various task samples."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or identifier for the model to load (e.g., 'Polygl0t/Tucano2-qwen-0.5B-Instruct')"
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        default=None,
        help="Path to a JSON file containing test samples. If not provided, uses hardcoded samples."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_task_outputs.json",
        help="Path to save the output results JSON file (default: model_task_outputs.json)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable 'thinking' mode in chat template."
    )
    return parser.parse_args()


def load_samples(samples_file: str = None) -> List[Dict[str, Any]]:
    """Load samples from file or use default hardcoded samples."""
    if samples_file is None:
        print("Using hardcoded default samples.")
        return DEFAULT_SAMPLES
    
    samples_path = Path(samples_file)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")
    
    print(f"Loading samples from: {samples_file}")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        raise ValueError("Samples file must contain a JSON array of sample objects.")
    
    print(f"Loaded {len(samples)} samples.")
    return samples


def clean_prompt(prompt: str) -> str:
    """Clean and format the prompt for display, preserving formatting."""
    # Escape # to prevent markdown heading interpretation
    prompt = prompt.replace('#', r'\#')
    return prompt.strip()


def clean_response(response: str, fix_encoding: bool = False) -> str:
    """Clean and format the response for display, optionally handling encoding issues."""
    if not fix_encoding:
        return response.strip()
    
    # Handle potential encoding issues only when explicitly requested
    if isinstance(response, bytes):
        response = response.decode('utf-8', errors='replace')
    
    # Try to fix common encoding issues (mojibake from latin1->utf8 double encoding)
    try:
        # Ensure proper UTF-8 encoding
        response = response.encode('latin1').decode('utf-8', errors='replace')
    except:
        # If conversion fails, just use the original with error replacement
        response = str(response).encode('utf-8', errors='replace').decode('utf-8')
    
    return response.strip()


def fix_json_encoding(json_str: str) -> str:
    """Fix double-escaped unicode characters in JSON string."""
    import json as json_module
    
    try:
        # Fix double-escaped unicode: \\u00e7 -> \u00e7
        json_str = json_str.replace('\\\\u', '\\u')
        
        # Parse and re-serialize to ensure proper formatting
        data = json_module.loads(json_str)
        return json_module.dumps(data, ensure_ascii=False, indent=2)
    except:
        # If JSON parsing fails, return original
        return json_str


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent text to create a code block in markdown without using backticks."""
    indent = ' ' * spaces
    lines = text.split('\n')
    return '\n'.join(indent + line for line in lines)


def generate_markdown_report(samples: list, output_path: str):
    """
    Generate a markdown report from inference samples.
    
    Args:
        samples: List of inference sample dictionaries
        output_path: Path to save the markdown report
    """
    
    # Start building the markdown content
    md_lines = []
    
    # Header
    md_lines.append("# Inference Samples Report\n")
    md_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n---\n")
    
    # Summary statistics
    md_lines.append("## Summary Statistics\n")
    
    task_types = {}
    has_eos_count = 0
    empty_output_count = 0
    total_tokens = 0
    
    for sample in samples:
        task_type = sample.get('task_type', 'Unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1
        
        if sample.get('checks', {}).get('has_eos', False):
            has_eos_count += 1
        if sample.get('checks', {}).get('empty_output', False):
            empty_output_count += 1
        
        total_tokens += sample.get('num_generated_tokens', 0)
    
    md_lines.append(f"- **Total Generated Tokens:** {total_tokens}")
    md_lines.append(f"- **Average Tokens per Sample:** {total_tokens / len(samples):.2f}")
    md_lines.append(f"- **Samples with EOS:** {has_eos_count} ({has_eos_count/len(samples)*100:.1f}%)")
    md_lines.append(f"- **Samples with Empty Output:** {empty_output_count} ({empty_output_count/len(samples)*100:.1f}%)")
    
    md_lines.append("\n---\n")
    
    # Detailed sample analysis
    md_lines.append("## Detailed Sample Analysis\n")
    
    for sample in samples:
        task_type = sample.get('task_type', 'Unknown')
        num_tokens = sample.get('num_generated_tokens', 0)
        has_eos = sample.get('checks', {}).get('has_eos', False)
        empty_output = sample.get('checks', {}).get('empty_output', False)
        
        md_lines.append(f"### {task_type}\n")
        
        # Status indicators
        status = []
        if has_eos:
            status.append("✅ Has EOS")
        else:
            status.append("❌ No EOS")
        
        if empty_output:
            status.append("⚠️ Empty Output")
        
        md_lines.append(f"**Status:** {' | '.join(status)} | **Tokens:** {num_tokens}")
        
        # Analysis notes (show issues upfront if any)
        notes = []
        if not has_eos:
            notes.append("⚠️ Missing EOS token")
        if num_tokens >= 1024:
            notes.append("📏 Hit max token limit")
        if empty_output:
            notes.append("❌ Empty output")
        
        if notes:
            md_lines.append(f" | {', '.join(notes)}")
        
        md_lines.append("\n")
        
        # Task-specific checks
        checks = sample.get('checks', {})
        task_specific_info = []
        
        if task_type == "Classification":
            if 'has_classification_label' in checks:
                label_status = "✅ Has label" if checks['has_classification_label'] else "❌ No label"
                task_specific_info.append(f"- **Classification Label Check:** {label_status}")
                if checks.get('classification_label'):
                    task_specific_info.append(f"- **Detected Label:** {checks['classification_label']}")
        
        elif task_type == "Structured Output":
            if 'valid_json' in checks:
                json_status = "✅ Valid" if checks['valid_json'] else "❌ Invalid"
                task_specific_info.append(f"- **JSON Validation:** {json_status}")
                if checks['valid_json'] and checks.get('json_output'):
                    task_specific_info.append(f"- **JSON Keys:** {list(checks['json_output'].keys())}")
        
        elif task_type == "Function Call / Tool Use":
            if 'made_function_call' in checks:
                call_status = "✅ Made function call" if checks['made_function_call'] else "❌ No function call"
                task_specific_info.append(f"- **Function Call:** {call_status}")
        
        elif task_type == "Summarization":
            if 'output_shorter_than_prompt' in checks:
                shorter_status = "✅ Yes" if checks['output_shorter_than_prompt'] else "❌ No"
                task_specific_info.append(f"- **Output Shorter than Prompt:** {shorter_status}")
            if 'prompt_length' in checks and 'output_length' in checks:
                task_specific_info.append(f"- **Prompt Length:** {checks['prompt_length']} chars")
                task_specific_info.append(f"- **Output Length:** {checks['output_length']} chars")
        
        elif "Reasoning" in task_type or task_type == "Math (Reasoning)":
            if 'has_think_tags' in checks:
                think_status = "✅ Yes" if checks['has_think_tags'] else "❌ No"
                task_specific_info.append(f"- **Has Think Tags:** {think_status}")
            if 'has_non_empty_think' in checks:
                non_empty_status = "✅ Yes" if checks['has_non_empty_think'] else "❌ No"
                task_specific_info.append(f"- **Non-empty Think Content:** {non_empty_status}")
            if 'think_tag_count' in checks:
                task_specific_info.append(f"- **Think Tag Count:** {checks['think_tag_count']}")
        
        if task_specific_info:
            md_lines.append("**Task-Specific Checks:**\n")
            md_lines.extend(task_specific_info)
            md_lines.append("\n")
        
        # Prompt
        prompt = clean_prompt(sample.get('prompt', ''))
        md_lines.append("**Prompt:**\n")
        # Indent the prompt to preserve formatting and prevent # from being headings
        md_lines.append(indent_text(prompt))
        md_lines.append("\n")
        
        # Generated response
        response = sample.get('generated_text', '')
        
        # Only apply encoding fixes for Structured Output tasks
        if task_type == "Structured Output":
            # Try to extract and fix JSON content
            try:
                # Find JSON content in response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_content = json_match.group(0)
                    fixed_json = fix_json_encoding(json_content)
                    response = response.replace(json_content, fixed_json)
            except:
                pass
        
        response = response.strip()
        
        # Truncate very long responses for readability
        if len(response) > 2000:
            md_lines.append(f"**Response:** *(truncated - showing first 2000 of {len(response)} characters)*\n")
            response = response[:2000] + "\n\n[... truncated ...]"
        else:
            md_lines.append("**Response:**\n")
        
        # Indent the response to preserve formatting
        md_lines.append(indent_text(response))
        md_lines.append("\n")
        
        md_lines.append("\n---\n")
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    return output_path



def main():
    args = parse_args()
    
    # Load samples
    samples = load_samples(args.samples_file)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.2,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Helper functions
    def format_chat(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, enable_thinking: bool = False) -> str:
        """Apply the model chat template safely."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        
        if tools:
            kwargs["tools"] = tools
        
        return tokenizer.apply_chat_template(
            messages,
            **kwargs
        )

    def generate_one(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, enable_thinking: bool = False) -> Dict[str, Any]:
        """Run a single generation and return text + metadata."""
        prompt = format_chat(messages, tools=tools, enable_thinking=enable_thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )

        full_sequence = outputs.sequences[0]
        generated_ids = full_sequence[len(inputs.input_ids[0]):]

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        return {
            "prompt": prompt,
            "generated_text": text,
            "generated_ids": generated_ids.tolist(),
            "eos_generated": tokenizer.eos_token_id in generated_ids.tolist(),
            "num_generated_tokens": len(generated_ids),
        }

    def run_task_checks(task_type: str, output: Dict[str, Any]) -> Dict[str, Any]:
        text = output["generated_text"]
        checks = {}

        # Generic checks
        checks["has_eos"] = output["eos_generated"]
        checks["empty_output"] = len(text.strip()) == 0

        # Classification check (e.g., sentiment classification)
        if task_type == "Classification":
            # Check if output contains expected classification labels
            has_positive = bool(re.search(r"\bpositivo\b", text.lower()))
            has_negative = bool(re.search(r"\bnegativo\b", text.lower()))
            has_neutral = bool(re.search(r"\bneutro\b", text.lower()))
            checks["has_classification_label"] = has_positive or has_negative or has_neutral
            checks["classification_label"] = "Positivo" if has_positive else ("Negativo" if has_negative else ("Neutro" if has_neutral else None))

        # Structured JSON output
        if task_type == "Structured Output":
            try:
                json_string = json_repair.repair_json(text)
                checks["valid_json"] = True
                checks["json_output"] = json_repair.loads(json_string)
            except Exception:
                checks["valid_json"] = False
                checks["json_output"] = None
        
        # Function Call / Tool Use
        if task_type == "Function Call / Tool Use":
            # Check if reply is a function call (i.e., <tool_call> ... </tool_call>)
            if re.search(r"<tool_call>.*</tool_call>", text, re.DOTALL):
                checks["made_function_call"] = True
            else:
                checks["made_function_call"] = False
        
        # Summarization check
        if task_type == "Summarization":
            prompt_length = len(output.get("prompt", ""))
            output_length = len(text)
            checks["output_shorter_than_prompt"] = output_length < prompt_length
            checks["prompt_length"] = prompt_length
            checks["output_length"] = output_length
        
        # Reasoning check - look for non-empty <think> tags
        if "Reasoning" in task_type:
            think_pattern = r"<think>(.*?)</think>"
            think_matches = re.findall(think_pattern, text, re.DOTALL)
            if think_matches:
                # Check if any of the think tags have non-empty content
                checks["has_think_tags"] = True
                checks["has_non_empty_think"] = any(match.strip() for match in think_matches)
                checks["think_tag_count"] = len(think_matches)
            else:
                checks["has_think_tags"] = False
                checks["has_non_empty_think"] = False
                checks["think_tag_count"] = 0

        return checks

    # Main evaluation loop
    results = []
    print(f"\nRunning inference on {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        task_type = sample.get("task_type", "unknown")
        print(f"\n=== Running sample {idx} | task={task_type} ===")

        # Extract tools if they exist
        tools = sample.get("tool", [])
        if tools:
            print(f"Using {len(tools)} tool(s)")
        
        # Enable thinking for reasoning tasks (or if globally enabled)
        is_reasoning_task = "reasoning" in task_type.lower()
        enable_thinking = args.enable_thinking or is_reasoning_task
        
        if enable_thinking:
            print(f"Thinking mode enabled for this task")
        
        generation = generate_one(sample["messages"], tools=tools if tools else None, enable_thinking=enable_thinking)
        checks = run_task_checks(task_type, generation)

        result = {
            "task_type": task_type,
            "prompt": generation["prompt"],
            "checks": checks,
            "generated_text": generation["generated_text"],
            "num_generated_tokens": generation["num_generated_tokens"],
            "has_eos": generation["eos_generated"],
        }

        results.append(result)

        print("Checks:", checks)
        print("Output:\n", generation["generated_text"])

    # Save results
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nAll tasks completed.")
    
    # Generate markdown report
    print("\nGenerating markdown analysis report...")
    output_path = Path(args.output_file)
    markdown_path = output_path.parent / f"{output_path.stem}.md"
    
    try:
        generate_markdown_report(results, str(markdown_path))
        print(f"✅ Markdown report saved to: {markdown_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate markdown report: {e}")


if __name__ == "__main__":
    main()
