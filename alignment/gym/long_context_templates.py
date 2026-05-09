"""
Long context retrieval task templates.

Templates for both word-list retrieval tasks and needle-in-a-haystack tasks.
Each template provides multiple preamble, question, and answer prefix variants
to ensure linguistic diversity in the generated prompts.

Word-list task types:
    - common_words:          Find the top-K most frequent words.
    - rare_words:            Find the top-K least frequent words.
    - count_word:            Count occurrences of a specific word.
    - word_at_position:      Identify the word at a numbered position.
    - frequency_comparison:  Compare frequency of two given words.

Haystack (needle-in-a-haystack) task types:
    - needle_single_number:          One number hidden in document for one key.
    - needle_multi_number_same_key:  Multiple numbers for the same key in document.
    - needle_multi_number_diff_keys: Numbers for different keys interleaved in document.
    - needle_uuid:                   UUID key-value pairs embedded in/around document text.

Template structure:
    id:              Unique task type identifier.
    task_type:       Logic key used by the generator for answer production.
    task_name:       Label stored in the output metadata.
    preambles:       Introductory texts placed before the context.
    questions:       Question format strings.
    needle_formats:  (haystack only) Format strings for needle sentences.
"""

LONG_CONTEXT_TEMPLATES = [
    #  Common Words (Top-K Most Frequent) 
    {
        "id": "common_words",
        "task_type": "common_words",
        "task_name": "long_context_common_words",
        "preambles": [
            "Abaixo está uma lista numerada de palavras. Nestas palavras, algumas aparecem com mais frequência do que outras. Memorize as que aparecem com mais frequência.",
            "A seguir, você verá uma lista longa de palavras numeradas. Algumas dessas palavras se repetem muitas vezes, enquanto outras aparecem poucas vezes. Preste atenção nas mais frequentes.",
            "Observe atentamente a lista de palavras numeradas abaixo. Certas palavras surgem repetidamente ao longo da lista. Identifique quais são as mais recorrentes.",
            "Você receberá uma lista numerada contendo diversas palavras. Nessa lista, há palavras que aparecem com alta frequência e outras com baixa frequência. Memorize as palavras mais comuns.",
            "Leia com atenção a lista de palavras a seguir. Algumas palavras aparecem muitas vezes e outras aparecem raramente. Concentre-se nas palavras que mais se repetem.",
            "Abaixo há uma extensa lista numerada de palavras. Entre elas, algumas são muito mais frequentes do que outras. Seu objetivo é identificar as mais comuns.",
            "Considere a seguinte lista numerada de palavras. Nela, determinadas palavras aparecem com muito mais frequência. Leia tudo e identifique as mais frequentes.",
            "Examine a lista de palavras numeradas apresentada abaixo. Algumas palavras foram inseridas múltiplas vezes. Descubra quais são as que mais aparecem.",
        ],
        "questions": [
            "Quais são as {top_k} palavras mais comuns na lista acima?",
            "Identifique as {top_k} palavras que aparecem com maior frequência na lista.",
            "Liste as {top_k} palavras mais frequentes da lista acima.",
            "Quais {top_k} palavras se repetem com mais frequência na lista?",
            "Dentre todas as palavras acima, quais são as {top_k} mais recorrentes?",
            "Aponte as {top_k} palavras que surgem o maior número de vezes na lista.",
            "Diga quais são as {top_k} palavras com maior número de ocorrências.",
            "Das palavras listadas, quais {top_k} possuem a maior frequência?",
        ],
    },
    #  Rare Words (Least Frequent) 
    {
        "id": "rare_words",
        "task_type": "rare_words",
        "task_name": "long_context_rare_words",
        "preambles": [
            "Abaixo está uma lista numerada de palavras. Algumas aparecem muitas vezes e outras aparecem poucas vezes. Preste atenção nas palavras que aparecem menos.",
            "A seguir, há uma longa lista de palavras numeradas. Nem todas as palavras têm a mesma frequência. Identifique as que aparecem com menor frequência.",
            "Observe a lista numerada de palavras abaixo. Algumas palavras são raras e aparecem poucas vezes. Concentre-se nessas palavras incomuns.",
            "Você verá uma lista de palavras onde algumas são muito comuns e outras bastante raras. Memorize as palavras que aparecem menos vezes.",
            "Leia a lista de palavras a seguir com atenção. Entre todas as palavras, algumas aparecem muito raramente. Identifique as menos frequentes.",
            "Abaixo há uma lista extensa de palavras numeradas. Algumas se repetem com frequência, mas outras quase não aparecem. Preste atenção nestas últimas.",
            "Analise a seguinte lista numerada com cuidado. Algumas palavras ocorrem muitas vezes, porém outras aparecem apenas uma ou duas vezes. Encontre as mais raras.",
            "Examine a lista de palavras abaixo. Sua tarefa é localizar as palavras que aparecem com menor frequência entre todas as listadas.",
        ],
        "questions": [
            "Quais são as {top_k} palavras menos comuns na lista acima?",
            "Identifique as {top_k} palavras que aparecem com menor frequência na lista.",
            "Liste as {top_k} palavras mais raras da lista acima.",
            "Quais {top_k} palavras aparecem menos vezes na lista?",
            "Dentre todas as palavras acima, quais são as {top_k} menos frequentes?",
            "Aponte as {top_k} palavras que surgem o menor número de vezes na lista.",
            "Diga quais são as {top_k} palavras com menor número de aparições.",
            "Das palavras listadas, quais {top_k} são as menos recorrentes?",
        ],
    },
    #  Count Word (Exact Count of a Specific Word) 
    {
        "id": "count_word",
        "task_type": "count_word",
        "task_name": "long_context_count_word",
        "preambles": [
            "Abaixo está uma lista numerada de palavras. Algumas se repetem diversas vezes.",
            "A seguir, há uma extensa lista de palavras numeradas. Preste atenção a quantas vezes cada palavra aparece.",
            "Observe a lista numerada de palavras abaixo. Conte com cuidado as ocorrências de cada palavra.",
            "Você receberá uma lista longa de palavras. Algumas dessas palavras aparecem várias vezes ao longo da lista.",
            "Leia a lista de palavras a seguir. Sua tarefa será contar quantas vezes uma palavra específica aparece.",
            "Abaixo há uma lista numerada de palavras em que certas palavras são repetidas. Conte as ocorrências com atenção.",
            "Analise a lista de palavras a seguir. Ao longo dela, algumas palavras se repetem um número variado de vezes.",
            "Examine a lista numerada de palavras apresentada abaixo. Determine o número exato de vezes que uma determinada palavra aparece.",
        ],
        "questions": [
            "Quantas vezes a palavra \"{target_word}\" aparece na lista acima?",
            "Conte o número de ocorrências da palavra \"{target_word}\" na lista.",
            "Na lista acima, quantas vezes a palavra \"{target_word}\" foi mencionada?",
            "Qual é o total de vezes que \"{target_word}\" aparece na lista?",
            "Diga quantas vezes a palavra \"{target_word}\" surge na lista acima.",
            "Quantas ocorrências da palavra \"{target_word}\" existem na lista?",
            "A palavra \"{target_word}\" aparece quantas vezes na lista acima?",
            "Informe o número exato de vezes que \"{target_word}\" aparece na lista.",
        ],
    },
    #  Word at Position 
    {
        "id": "word_at_position",
        "task_type": "word_at_position",
        "task_name": "long_context_word_at_position",
        "preambles": [
            "Abaixo está uma lista numerada de palavras.",
            "A seguir, há uma lista de palavras organizada por números sequenciais.",
            "Observe a lista numerada de palavras abaixo. Cada palavra está associada a uma posição numérica.",
            "Você receberá uma lista de palavras numeradas sequencialmente.",
            "Leia a lista numerada de palavras a seguir. Cada posição contém uma palavra.",
            "Abaixo há uma lista de palavras, cada uma com um número identificador de posição.",
            "Considere a lista de palavras numeradas apresentada abaixo. Cada número corresponde a uma palavra específica.",
            "Examine a lista a seguir. As palavras estão numeradas em sequência, da primeira até a última.",
        ],
        "questions": [
            "Qual palavra está na posição {position} da lista?",
            "Que palavra aparece na posição número {position}?",
            "Identifique a palavra que ocupa a posição {position} na lista.",
            "Na lista acima, qual é a palavra na posição {position}?",
            "Diga qual palavra se encontra na posição {position} da lista acima.",
            "Qual é a palavra de número {position} na lista?",
            "Que palavra ocupa o lugar de número {position} na lista?",
            "Informe a palavra correspondente à posição {position}.",
        ],
    },
    #  Frequency Comparison 
    {
        "id": "frequency_comparison",
        "task_type": "frequency_comparison",
        "task_name": "long_context_frequency_comparison",
        "preambles": [
            "Abaixo está uma lista numerada de palavras, onde algumas aparecem com frequências diferentes.",
            "A seguir, há uma lista longa de palavras numeradas. Algumas palavras se repetem mais do que outras.",
            "Observe a lista de palavras numeradas abaixo. As palavras têm frequências de aparição variadas.",
            "Você receberá uma lista de palavras onde cada uma aparece um número diferente de vezes.",
            "Leia a lista de palavras a seguir, prestando atenção à frequência com que cada uma aparece.",
            "Abaixo há uma lista numerada de palavras com diferentes níveis de repetição.",
            "Analise a lista numerada de palavras a seguir. Algumas palavras são mais frequentes do que outras.",
            "Examine a lista de palavras abaixo. Determine qual das palavras indicadas aparece mais vezes.",
        ],
        "questions": [
            "Entre as palavras \"{word_a}\" e \"{word_b}\", qual aparece mais vezes na lista?",
            "Comparando \"{word_a}\" e \"{word_b}\", qual é mais frequente na lista acima?",
            "Qual das duas palavras aparece com maior frequência na lista: \"{word_a}\" ou \"{word_b}\"?",
            "Na lista acima, qual palavra é mais comum: \"{word_a}\" ou \"{word_b}\"?",
            "Dentre \"{word_a}\" e \"{word_b}\", qual tem mais ocorrências na lista?",
            "Qual palavra aparece mais na lista: \"{word_a}\" ou \"{word_b}\"?",
            "Das duas palavras, \"{word_a}\" e \"{word_b}\", qual surge com mais frequência?",
            "Entre \"{word_a}\" e \"{word_b}\", qual foi mais repetida na lista acima?",
        ],
    },
    #  needle_single_number 
    {
        "id": "needle_single_number",
        "task_type": "needle_single_number",
        "task_name": "haystack_single_number",
        "preambles": [
            "Um número especial está escondido no texto abaixo. Memorize-o. Vou questioná-lo sobre ele depois.",
            "No texto a seguir, um número especial foi inserido. Leia com atenção e memorize-o. Vou perguntar sobre ele.",
            "Há um número especial oculto no texto abaixo. Preste atenção e memorize-o para responder à pergunta depois.",
            "Leia atentamente o texto a seguir. Um número especial foi escondido nele. Memorize-o.",
            "Um número especial foi disfarçado no meio do texto abaixo. Encontre-o e memorize-o.",
            "O texto abaixo contém um número especial escondido. Leia tudo e memorize-o para responder depois.",
        ],
        "needle_formats": [
            "Um número especial para {key} é: {value}.",
            "O número especial para {key} é: {value}.",
            "O número secreto para {key} é: {value}.",
        ],
        "questions": [
            "Qual é o número especial para {key} mencionado no texto?",
            "Qual é o número especial para {key} que aparece no texto acima?",
            "Diga qual é o número especial para {key} contido no texto.",
            "Identifique o número especial para {key} no texto fornecido.",
            "Que número especial para {key} foi mencionado no texto?",
            "Encontre o número especial para {key} no texto acima.",
        ],
    },
    #  needle_multi_number_same_key 
    {
        "id": "needle_multi_number_same_key",
        "task_type": "needle_multi_number_same_key",
        "task_name": "haystack_multi_number_same_key",
        "preambles": [
            "Alguns números especiais estão escondidos no texto abaixo. Memorize-os. Vou questioná-lo sobre eles depois.",
            "No texto a seguir, vários números especiais foram inseridos. Leia com atenção e memorize-os.",
            "Diversos números especiais estão ocultos no texto abaixo. Preste atenção a todos eles.",
            "Leia atentamente o texto a seguir. Múltiplos números especiais estão escondidos nele. Memorize-os.",
            "Vários números especiais foram disfarçados no texto abaixo. Encontre-os e memorize-os.",
            "O texto abaixo contém múltiplos números especiais escondidos. Leia tudo e memorize-os para responder depois.",
        ],
        "needle_formats": [
            "Um número especial para {key} é: {value}.",
            "O número especial para {key} é: {value}.",
            "O número secreto para {key} é: {value}.",
        ],
        "questions": [
            "Quais são todos os números especiais para {key} mencionados no texto?",
            "Liste todos os números especiais para {key} que aparecem no texto acima.",
            "Identifique todos os números especiais para {key} no texto fornecido.",
            "Quais números especiais para {key} foram mencionados no texto?",
            "Encontre todos os números especiais para {key} no texto acima.",
            "Diga quais são todos os números especiais para {key} contidos no texto.",
        ],
    },
    #  needle_multi_number_diff_keys 
    {
        "id": "needle_multi_number_diff_keys",
        "task_type": "needle_multi_number_diff_keys",
        "task_name": "haystack_multi_number_diff_keys",
        "preambles": [
            "Alguns números especiais estão escondidos no texto abaixo. Memorize-os. Vou questioná-lo sobre eles depois.",
            "No texto a seguir, números especiais para diferentes chaves foram inseridos. Leia com atenção e memorize-os.",
            "Diversos números especiais, cada um com uma chave diferente, estão ocultos no texto abaixo. Preste atenção.",
            "Leia atentamente o texto a seguir. Números especiais com chaves distintas estão escondidos nele.",
            "Vários números especiais com chaves diferentes foram disfarçados no texto abaixo. Encontre-os e memorize-os.",
            "O texto abaixo contém números especiais com chaves variadas. Leia tudo e memorize-os para responder depois.",
        ],
        "needle_formats": [
            "Um número especial para {key} é: {value}.",
            "O número especial para {key} é: {value}.",
            "O número secreto para {key} é: {value}.",
        ],
        "questions": [
            "Quais são os números especiais para {keys_str} mencionados no texto?",
            "Liste os números especiais para {keys_str} que aparecem no texto acima.",
            "Identifique os números especiais para {keys_str} no texto fornecido.",
            "Quais números especiais para {keys_str} foram mencionados no texto?",
            "Encontre os números especiais para {keys_str} no texto acima.",
            "Diga quais são os números especiais para {keys_str} contidos no texto.",
        ],
    },
    #  needle_uuid 
    {
        "id": "needle_uuid",
        "task_type": "needle_uuid",
        "task_name": "haystack_uuid",
        "preambles": [
            "Um código UUID especial está escondido no texto abaixo. Memorize-o. Vou questioná-lo sobre ele depois.",
            "No texto a seguir, códigos UUID especiais foram inseridos. Leia com atenção e memorize-os.",
            "Códigos UUID especiais estão ocultos no texto abaixo. Preste atenção e memorize-os.",
            "Leia atentamente o texto a seguir. Códigos UUID especiais estão escondidos nele. Memorize-os.",
            "Vários códigos UUID especiais foram disfarçados no texto abaixo. Encontre-os e memorize-os.",
            "O texto abaixo contém códigos UUID especiais. Leia tudo e memorize-os para responder depois.",
        ],
        "needle_formats": [
            "Um código UUID especial para {key} é: {value}.",
            "O código UUID especial para {key} é: {value}.",
            "O código UUID secreto para {key} é: {value}.",
        ],
        "questions": [
            "Qual é o código UUID especial para {query_key} mencionado no texto?",
            "Qual é o código UUID especial para {query_key} que aparece no texto acima?",
            "Diga qual é o código UUID especial para {query_key} contido no texto.",
            "Identifique o código UUID especial para {query_key} no texto fornecido.",
            "Que código UUID especial para {query_key} foi mencionado no texto?",
            "Encontre o código UUID especial para {query_key} no texto acima.",
        ],
    },
]

# All available task types (convenience list)
LONG_CONTEXT_TASK_TYPES = [t["task_type"] for t in LONG_CONTEXT_TEMPLATES]

# Category sublists
WORD_LIST_TASK_TYPES = [
    "common_words", "rare_words", "count_word",
    "word_at_position", "frequency_comparison",
]
HAYSTACK_TASK_TYPES = [
    "needle_single_number", "needle_multi_number_same_key",
    "needle_multi_number_diff_keys", "needle_uuid",
]
