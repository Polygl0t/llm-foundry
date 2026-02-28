"""
Generate prompt variations for IFEval-PT dataset.

Usage:
python generate_variations.py \
    --input_file original_prompts.jsonl \
    --output_file all_prompts.jsonl \
    --num_variations 4
"""

import json
import re
import random
from pathlib import Path
import argparse

random.seed(42)

# Gender: 'm' = masculine (o/ao/do/no), 'f' = feminine (a/à/da/na), 'n' = no article (a/de/em)
COUNTRY_ALTERNATIVES = {
    'Brasil': ['Argentina', 'Chile', 'Colômbia'],
    'Japão': ['Coreia do Sul', 'Tailândia', 'Vietnã'],
    'China': ['Índia', 'Indonésia', 'Malásia'],
    'Índia': ['China', 'Egito', 'Turquia'],
    'Portugal': ['Espanha', 'Itália', 'Grécia'],
    'Havaí': ['Bahamas', 'Maldivas', 'Ilha de Páscoa'],
    'Suíça': ['Áustria', 'Noruega', 'Dinamarca'],
    'Reino Unido': ['França', 'Alemanha', 'Itália'],
    'Argentina': ['Chile', 'Uruguai', 'Peru'],
    'Terra': ['Marte', 'nosso planeta', 'o mundo'],
    'Moçambique': ['Angola', 'Cabo Verde', 'Guiné-Bissau'],
}

ENTITY_GENDER = {
    # Countries - masculine
    'Brasil': 'm', 'Japão': 'm', 'Chile': 'm', 'Peru': 'm', 'México': 'm',
    'Egito': 'm', 'Vietnã': 'm', 'Havaí': 'm', 'Uruguai': 'm', 'Nepal': 'm',
    'Paquistão': 'm', 'Reino Unido': 'm', 'Equador': 'm',
    # Countries - feminine
    'Argentina': 'f', 'Colômbia': 'f', 'China': 'f', 'Índia': 'f',
    'Coreia do Sul': 'f', 'Tailândia': 'f', 'Indonésia': 'f', 'Malásia': 'f',
    'Turquia': 'f', 'Espanha': 'f', 'Itália': 'f', 'Grécia': 'f',
    'França': 'f', 'Alemanha': 'f', 'Áustria': 'f', 'Noruega': 'f',
    'Dinamarca': 'f', 'Suíça': 'f', 'Terra': 'f',
    'Bahamas': 'f', 'Maldivas': 'f', 'Ilha de Páscoa': 'f',
    # Countries - no article
    'Portugal': 'n', 'Moçambique': 'n', 'Angola': 'n', 'Cabo Verde': 'n',
    'Guiné-Bissau': 'n', 'Marte': 'n',
    # Topics - masculine
    'café': 'm', 'chá': 'm', 'vinho': 'm', 'cachorro': 'm', 'gato': 'm',
    'coelho': 'm', 'hamster': 'm', 'patinete': 'm', 'skate': 'm',
    'colchão': 'm', 'travesseiro': 'm', 'bolo': 'm', 'brigadeiro': 'm',
    'pão de queijo': 'm', 'tigre': 'm', 'violão': 'm', 'piano': 'm',
    'tablet': 'm', 'notebook': 'm', 'plástico': 'm', 'vidro': 'm',
    'sorvete': 'm', 'smartphone': 'm', 'alumínio': 'm', 'churrasco': 'm',
    'bordado': 'm', 'tricô': 'm', 'crochê': 'm', 'ipê': 'm',
    'girassol': 'm', 'leão': 'm', 'koala': 'm', 'urso polar': 'm',
    'roncar': 'm', 'misto quente': 'm',
    # Topics - feminine
    'rede': 'f', 'fralda': 'f', 'mamadeira': 'f', 'chupeta': 'f',
    'rosa': 'f', 'orquídea': 'f', 'tapioca': 'f', 'feijoada': 'f',
    'moqueca': 'f', 'bicicleta': 'f', 'cerveja': 'f', 'banana': 'f',
    'maçã': 'f', 'laranja': 'f', 'araucária': 'f', 'internet': 'f',
    'harpa': 'f', 'ferrugem': 'f', 'sujeira': 'f', 'mancha': 'f',
    'águia': 'f', 'insônia': 'f',
}

CITY_ALTERNATIVES = {
    'São Paulo': ['Rio de Janeiro', 'Belo Horizonte', 'Curitiba'],
    'Porto Alegre': ['Florianópolis', 'Curitiba', 'Gramado'],
    'Fortaleza': ['Recife', 'Salvador', 'Natal'],
    'Florianópolis': ['Curitiba', 'Porto Alegre', 'Joinville'],
    'Moscou': ['Berlim', 'Viena', 'Praga'],
    'Londres': ['Paris', 'Berlim', 'Roma'],
    'Helsinki': ['Estocolmo', 'Oslo', 'Copenhague'],
    'Pequim': ['Tóquio', 'Seul', 'Bangkok'],
    'Juiz de Fora': ['Ouro Preto', 'Mariana', 'Tiradentes'],
    'Alvorada': ['Canoas', 'Viamão', 'Gravataí'],
    'Tulsa': ['Austin', 'Denver', 'Nashville'],
    'Nova York': ['Chicago', 'Los Angeles', 'Boston'],
    'Saskatoon': ['Vancouver', 'Toronto', 'Winnipeg'],
    'Copa Cabana': ['Ipanema', 'Leblon', 'Barra da Tijuca'],
}

NAME_ALTERNATIVES = {
    'Matthias Algiers': ['Carlos Mendes', 'Roberto Farias'],
    'Antonia Maj': ['Carolina Silva', 'Fernanda Costa'],
    'Naomi': ['Daniela', 'Mariana'],
    'Clarissa': ['Fernanda', 'Juliana'],
    'Jean': ['Paulo', 'André'],
    'Bozo': ['Patati', 'Carequinha'],
    'Camila': ['Mariana', 'Rafaela'],
    'Tereza': ['Helena', 'Cecília'],
    'Susana': ['Renata', 'Patrícia'],
    'Gilson': ['Arnaldo', 'Benedito'],
    'Jonas': ['Marcos', 'Ícaro'],
    'Luheng': ['Kenji', 'Yuki'],
    'Pitoco': ['Bolinha', 'Faísca'],
    'Jennifer': ['Roberta', 'Fernanda'],
    'Vitor': ['Marcos', 'Gustavo'],
    'Breno': ['Lucas', 'Gabriel'],
    'João': ['Pedro', 'Carlos'],
    'Alexandre': ['Júlio César', 'Gengis Khan'],
}

TOPIC_KEYWORD_SWAPS = {
    'café': ['chá', 'vinho'],
    'internet': ['inteligência artificial', 'redes sociais'],
    'dinossauros': ['mamutes', 'criaturas pré-históricas'],
    'cachorro': ['gato', 'coelho'],
    'gato': ['cachorro', 'hamster'],
    'bicicleta': ['patinete', 'skate'],
    'foguetes': ['satélites', 'aviões'],
    'misto quente': ['pão de queijo', 'tapioca'],
    'sorvete': ['bolo', 'brigadeiro'],
    'rede': ['colchão', 'travesseiro'],
    'churrasco': ['feijoada', 'moqueca'],
    'tomates': ['cebolas', 'abóboras'],
    'sapatos': ['bolsas', 'acessórios'],
    'miniaturas': ['maquetes', 'réplicas'],
    'eucalipto': ['ipê', 'araucária'],
    'bordado': ['tricô', 'crochê'],
    'rosa': ['girassol', 'orquídea'],
    'legumes': ['frutas', 'verduras'],
    'xadrez': ['damas', 'gamão'],
    'leão': ['tigre', 'águia'],
    'panda': ['koala', 'urso polar'],
    'baleia': ['tubarão', 'golfinho'],
    'piano': ['violão', 'harpa'],
    'corgi': ['labrador', 'poodle'],
    'fralda': ['mamadeira', 'chupeta'],
    'cerveja': ['refrigerante', 'suco'],
    'ferrugem': ['sujeira', 'mancha'],
    'abismo': ['vulcão', 'oceano'],
    'smartphone': ['tablet', 'notebook'],
    'alumínio': ['plástico', 'vidro'],
    'banana': ['maçã', 'laranja'],
    'vermes': ['insetos', 'aranhas'],
}


def adjust_number(n, direction='up'):
    """Adjust a number up or down, keeping it reasonable."""
    if direction == 'up':
        if n <= 2:
            return n + random.choice([1, 2])
        elif n <= 5:
            return n + random.choice([1, 2, 3])
        elif n <= 10:
            return n + random.choice([2, 3, 4, 5])
        elif n <= 30:
            return n + random.choice([5, 8, 10, 12])
        elif n <= 100:
            return n + random.choice([20, 30, 50])
        elif n <= 500:
            return n + random.choice([50, 100, 150, 200])
        else:
            return int(n * random.uniform(1.3, 1.7))
    else:  # down
        if n <= 2:
            return max(1, n)
        elif n <= 5:
            return max(1, n - random.choice([1, 2]))
        elif n <= 10:
            return max(2, n - random.choice([1, 2, 3]))
        elif n <= 30:
            return max(3, n - random.choice([3, 5, 8, 10]))
        elif n <= 100:
            return max(10, n - random.choice([10, 20, 30]))
        elif n <= 500:
            return max(50, n - random.choice([50, 80, 100, 150]))
        else:
            return max(100, int(n * random.uniform(0.5, 0.7)))


def is_likely_year(n):
    """Check if a number is likely a year."""
    return 1800 <= n <= 2100


def is_likely_constraint_number(text, match_start, match_end):
    """Check if a number in the text is likely a constraint number."""
    context_before = text[max(0, match_start - 50):match_start].lower()
    context_after = text[match_end:match_end + 50].lower()
    
    constraint_before = any(w in context_before for w in [
        'pelo menos', 'no máximo', 'exatamente', 'mais de', 'menos de',
        'entre', 'mínimo', 'máximo', 'conter', 'incluir',
        'com', 'ter', 'aparecer', 'apareça'
    ])
    
    constraint_after = any(w in context_after for w in [
        'palavras', 'palavra', 'frases', 'frase', 'parágrafos', 'parágrafo',
        'seções', 'seção', 'vezes', 'vez', 'pontos', 'ponto',
        'itens', 'item', 'dias', 'dia', 'linhas', 'linha',
        'marcadores', 'marcador', 'estrofes', 'estrofe',
        'hashtags', 'hashtag', 'espaços reservados', 'espaço reservado',
        'caracteres', 'caractere', 'bullet', 'colchetes',
        'pontos de exclamação', 'asterisco', 'placeholders',
        'ou mais', 'ou menos',
    ])
    
    return constraint_before or constraint_after


def vary_constraint_numbers(text, direction='up'):
    """
    Find numbers in constraint contexts and adjust them.
    Returns the modified text.
    """
    # Pattern: number that's near constraint keywords
    result = []
    last_end = 0
    
    for match in re.finditer(r'\b(\d+)\b', text):
        n = int(match.group(1))
        start, end = match.start(), match.end()
        
        # Skip years
        if is_likely_year(n):
            continue
        
        # Skip phone numbers (digits near hyphens with other digits)
        surrounding = text[max(0, start - 5):end + 5]
        if re.search(r'\d+-\d+', surrounding):
            continue
        
        # Only modify constraint numbers
        if is_likely_constraint_number(text, start, end):
            new_n = adjust_number(n, direction)
            result.append(text[last_end:start])
            result.append(str(new_n))
            last_end = end
    
    result.append(text[last_end:])
    return ''.join(result)

def swap_quantifier(text, variation):
    """Swap quantifiers like 'pelo menos' <-> 'no máximo' etc."""
    if variation == 1:
        # "pelo menos" -> "no mínimo" (similar but different wording)
        text = re.sub(r'\bpelo menos\b', 'no mínimo', text, count=1)
        text = re.sub(r'\bmais de\b', 'pelo menos', text, count=1)
    else:
        # "pelo menos" -> "mais de" (slightly different meaning)
        text = re.sub(r'\bpelo menos\b', 'mais de', text, count=1)
        # "exatamente" -> "pelo menos" (relaxing constraint)
        text = re.sub(r'\bexatamente\b', 'pelo menos', text, count=1)
    return text

def fix_preposition_gender(text, original_entity, new_entity):
    """
    Fix Portuguese preposition+article forms when substituting entities.
    E.g., 'ao Japão' → 'à Coreia do Sul' (m→f), 'do Brasil' → 'da Argentina' (m→f)
    """
    orig_gender = ENTITY_GENDER.get(original_entity, 'n')
    new_gender = ENTITY_GENDER.get(new_entity, 'n')
    
    if orig_gender == new_gender:
        return text
    
    # Preposition mappings: (before_pattern, m_form, f_form, n_form)
    preposition_forms = [
        (r'\bao\s+' + re.escape(new_entity), 'ao ' + new_entity, 'à ' + new_entity, 'a ' + new_entity),
        (r'\bà\s+' + re.escape(new_entity), 'ao ' + new_entity, 'à ' + new_entity, 'a ' + new_entity),
        (r'\ba\s+' + re.escape(new_entity), 'ao ' + new_entity, 'à ' + new_entity, 'a ' + new_entity),
        (r'\bdo\s+' + re.escape(new_entity), 'do ' + new_entity, 'da ' + new_entity, 'de ' + new_entity),
        (r'\bda\s+' + re.escape(new_entity), 'do ' + new_entity, 'da ' + new_entity, 'de ' + new_entity),
        (r'\bde\s+' + re.escape(new_entity), 'do ' + new_entity, 'da ' + new_entity, 'de ' + new_entity),
        (r'\bno\s+' + re.escape(new_entity), 'no ' + new_entity, 'na ' + new_entity, 'em ' + new_entity),
        (r'\bna\s+' + re.escape(new_entity), 'no ' + new_entity, 'na ' + new_entity, 'em ' + new_entity),
        (r'\bem\s+' + re.escape(new_entity), 'no ' + new_entity, 'na ' + new_entity, 'em ' + new_entity),
        (r'\bo\s+' + re.escape(new_entity), 'o ' + new_entity, 'a ' + new_entity, '' + new_entity),
    ]
    
    # Also handle with um/uma
    preposition_forms += [
        (r'\bum\s+' + re.escape(new_entity), 'um ' + new_entity, 'uma ' + new_entity, 'um ' + new_entity),
        (r'\buma\s+' + re.escape(new_entity), 'um ' + new_entity, 'uma ' + new_entity, 'uma ' + new_entity),
    ]
    
    for pattern, m_form, f_form, n_form in preposition_forms:
        if new_gender == 'm':
            replacement = m_form
        elif new_gender == 'f':
            replacement = f_form
        else:
            replacement = n_form
        text = re.sub(pattern, replacement, text)
    
    return text


def substitute_entities(text, variation_idx=0):
    """
    Substitute names, places, and topics with gender-aware preposition fixing.
    variation_idx controls which alternative to pick (0 or 1).
    """
    result = text
    
    # Sort by length descending to avoid partial matches
    all_subs = {}
    all_subs.update(NAME_ALTERNATIVES)
    all_subs.update(CITY_ALTERNATIVES)
    all_subs.update(COUNTRY_ALTERNATIVES)
    
    sorted_keys = sorted(all_subs.keys(), key=len, reverse=True)
    
    for original in sorted_keys:
        if original in result:
            alternatives = all_subs[original]
            idx = min(variation_idx, len(alternatives) - 1)
            replacement = alternatives[idx]
            result = result.replace(original, replacement)
            # Fix preposition/article gender agreement
            result = fix_preposition_gender(result, original, replacement)
    
    return result


def substitute_topics(text, variation_idx=0):
    """Substitute topic keywords with gender-aware article fixing."""
    result = text
    
    sorted_keys = sorted(TOPIC_KEYWORD_SWAPS.keys(), key=len, reverse=True)
    
    for original in sorted_keys:
        # Use word boundary matching for topic keywords
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        if pattern.search(result):
            alternatives = TOPIC_KEYWORD_SWAPS[original]
            idx = min(variation_idx, len(alternatives) - 1)
            replacement = alternatives[idx]
            # Preserve original case of first letter
            def case_preserving_replace(match, rep=replacement):
                orig = match.group()
                if orig[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep
            result = pattern.sub(case_preserving_replace, result, count=1)
            # Fix article/preposition gender
            result = fix_preposition_gender(result, original, replacement)
            break  # Only substitute one topic per call to avoid cascading issues
    
    return result

ADDITIONAL_CONSTRAINTS_PT = [
    'Não use vírgulas na sua resposta.',
    'Sua resposta deve conter pelo menos {n} frases.',
    'Certifique-se de que a letra \'{letter}\' apareça pelo menos {n} vezes na sua resposta.',
    'Inclua a palavra-chave \'{keyword}\' pelo menos {n} vezes na sua resposta.',
    'Adicione uma nota pós-escrita começando com "P.S." no final da sua resposta.',
    'Destaque pelo menos {n} seções usando markdown, como *seção destacada*.',
    'Sua resposta deve estar toda em letras minúsculas.',
    'Envolva toda a sua resposta entre aspas duplas.',
    'Inclua pelo menos {n} espaços reservados representados por colchetes, como [nome].',
    'Termine sua resposta com a frase exata: "Há algo mais em que eu possa ajudar?"',
    'Inclua um título entre duplas setas angulares, como <<título>>.',
    'Use menos de {n} frases na sua resposta.',
    'A resposta completa deve ter entre {n1} e {n2} palavras.',
    'Sua resposta deve ter no mínimo {n1} e no máximo {n2} palavras.',
]

KEYWORDS_PT = [
    'fundamental', 'consequência', 'perspectiva', 'inovação', 'sustentável',
    'paradigma', 'contexto', 'relevante', 'impacto', 'estratégia',
    'primordial', 'essencial', 'significativo', 'correlação', 'evidência',
    'tecnologia', 'diversidade', 'criatividade', 'transformação', 'evolução',
]

LETTERS_PT = ['a', 'e', 'o', 'i', 'r', 's', 'n', 'l', 'p', 'q']


def add_random_constraint(text):
    """Add a random constraint to a prompt."""
    constraint_template = random.choice(ADDITIONAL_CONSTRAINTS_PT)
    
    # Fill in template variables
    constraint = constraint_template.format(
        n=random.choice([2, 3, 4, 5, 6, 8, 10]),
        n1=random.choice([100, 150, 200, 300, 400]),
        n2=random.choice([250, 300, 400, 500, 600, 700]),
        letter=random.choice(LETTERS_PT),
        keyword=random.choice(KEYWORDS_PT),
    )
    
    # Don't add constraints that conflict with existing ones
    # Check for obvious conflicts
    text_lower = text.lower()
    if 'letras minúsculas' in text_lower and 'letras minúsculas' in constraint.lower():
        return text
    if 'letras maiúsculas' in text_lower and 'letras minúsculas' in constraint.lower():
        return text
    if 'aspas duplas' in text_lower and 'aspas duplas' in constraint.lower():
        return text
    if 'vírgulas' in text_lower and 'vírgulas' in constraint.lower():
        return text
    if 'p.s.' in text_lower and 'P.S.' in constraint:
        return text
    if 'p.p.s' in text_lower and 'P.S.' in constraint:
        return text
    if '<<' in text and '<<' in constraint:
        return text
    
    # Append the constraint
    if text.endswith('.') or text.endswith('。') or text.endswith('"'):
        text = text + ' ' + constraint
    else:
        text = text + '. ' + constraint
    
    return text


def modify_existing_constraint(text):
    """Modify an existing constraint in the text (avoiding contradictions)."""
    result = text
    lower = result.lower()
    
    # Swap case constraints ONLY if it won't create contradiction
    # Count occurrences to detect "apenas maiúsculas, sem minúsculas" patterns
    has_maiusculas = 'letras maiúsculas' in lower
    has_minusculas = 'letras minúsculas' in lower
    has_sem_minusculas = 'sem letras minúsculas' in lower
    has_sem_maiusculas = 'sem letras maiúsculas' in lower
    has_apenas_maiusculas = 'apenas letras maiúsculas' in lower or 'todas as letras maiúsculas' in lower or 'toda em letras maiúsculas' in lower
    has_apenas_minusculas = 'apenas letras minúsculas' in lower or 'todas as letras minúsculas' in lower or 'toda em letras minúsculas' in lower or 'todas apenas letras minúsculas' in lower
    
    # Only do case swap if there's a single, simple case constraint
    if has_apenas_minusculas and not has_maiusculas:
        # Safe to swap: just a simple "all lowercase" constraint
        result = result.replace('letras minúsculas', 'letras maiúsculas')
    elif has_apenas_maiusculas and not has_minusculas and not has_sem_minusculas:
        # Has both "maiúsculas" and "sem minúsculas" - swap both
        result = result.replace('letras maiúsculas', 'letras minúsculas')
    elif has_minusculas and not has_maiusculas:
        result = result.replace('letras minúsculas', 'letras maiúsculas', 1)
    # Otherwise don't touch case constraints - too risky for contradictions
    
    # Swap separator style (only simple cases)
    if '***' in result and '******' not in result and '---' not in result:
        result = result.replace('***', '---')
    
    # Swap JSON <-> table
    if 'formato JSON' in result or 'formato json' in result:
        result = re.sub(r'formato JSON', 'formato de tabela', result, flags=re.IGNORECASE, count=1)
    elif 'formato de tabela' in lower:
        result = re.sub(r'formato de tabela', 'formato JSON', result, flags=re.IGNORECASE, count=1)
    
    return result

def swap_ending_phrase(text):
    """Swap the ending phrase requirement if present."""
    ending_phrases = {
        '"Há algo mais em que eu possa ajudar?"': '"Posso ajudar com algo mais?"',
        '"Há algo mais que eu possa ajudar?"': '"Posso ser útil em algo mais?"',
        '"Há algo mais que eu posso ajudar com?"': '"Me diga se posso ajudar com mais alguma coisa."',
        '"Posso receber meu dinheiro de volta pelas aulas que perdi?"': '"Será que vale a pena continuar pagando por essas aulas?"',
    }
    
    for original, replacement in ending_phrases.items():
        if original in text:
            text = text.replace(original, replacement)
            break
    
    return text


def swap_separator(text):
    """Swap paragraph/section separators."""
    if '******' in text:
        # Change the number of asterisks
        text = text.replace('******', '****')
    elif '***' in text:
        text = text.replace('***', '---')
    return text


def create_variation(prompt, variation_idx, num_variations):
    """
    Create a single variation of a prompt.
    
    The strategy is parameterized by variation_idx (0-based) so that
    different indices produce different transformations:
      - Even indices: numbers UP, odd indices: numbers DOWN
      - Entity/topic substitution index cycles through available alternatives
      - Quantifier swap style alternates
      - Various stochastic transforms with variation-dependent probabilities
    """
    result = prompt
    
    # Determine strategy based on variation index
    direction = 'up' if variation_idx % 2 == 0 else 'down'
    entity_idx = variation_idx % 3  # cycle through up to 3 alternatives
    topic_idx = variation_idx % 2
    quantifier_style = (variation_idx % 2) + 1  # 1 or 2
    
    # 1. Adjust constraint numbers
    result = vary_constraint_numbers(result, direction=direction)
    
    # 2. Swap quantifiers
    result = swap_quantifier(result, variation=quantifier_style)
    
    # 3. Substitute entity names
    result = substitute_entities(result, variation_idx=entity_idx)
    
    # 4. Topic substitutions (more likely on odd indices)
    if variation_idx % 2 == 1 or random.random() < 0.4:
        result = substitute_topics(result, variation_idx=topic_idx)
    
    # 5. Modify existing constraints (more likely on higher indices)
    if random.random() < 0.3 + 0.1 * min(variation_idx, 4):
        result = modify_existing_constraint(result)
    
    # 6. Swap ending phrases
    if random.random() < 0.3 + 0.1 * (variation_idx % 3):
        result = swap_ending_phrase(result)
    
    # 7. Swap separators (occasionally)
    if random.random() < 0.2 + 0.1 * (variation_idx % 3):
        result = swap_separator(result)
    
    # 8. Add a random constraint (probability increases with index)
    add_prob = 0.2 + 0.1 * min(variation_idx, 5)
    if random.random() < add_prob:
        result = add_random_constraint(result)
    
    # Fallback cascade: ensure variation differs from original
    if result == prompt:
        result = add_random_constraint(result)
    
    if result == prompt:
        result = substitute_topics(result, variation_idx=entity_idx % 2)
    
    # Absolute fallback: deterministic constraint addition
    if result == prompt:
        safe_constraints = [
            ' Inclua um título entre duplas setas angulares, como <<título>>.',
            ' Adicione uma nota pós-escrita começando com "P.S." no final da sua resposta.',
            ' Termine sua resposta com a frase exata: "Há algo mais em que eu possa ajudar?"',
            ' Sua resposta deve conter pelo menos 15 frases.',
            ' Inclua pelo menos 3 espaços reservados representados por colchetes, como [nome].',
            ' Certifique-se de que a resposta tenha pelo menos 200 palavras.',
            ' Não inclua a letra "z" em nenhum lugar da sua resposta.',
            ' Sua resposta deve conter entre 150 e 250 palavras.',
            ' Escreva sua resposta como se estivesse explicando para uma criança de 10 anos.',
            ' Certifique-se de que a resposta tenha no máximo 20 frases.',
            ' Envolva toda a resposta entre aspas duplas.',
            ' Inclua pelo menos 2 exemplos concretos na sua resposta.',
            ' Adicione uma nota pós-escrita começando com "P.P.S" no final da sua resposta.',
        ]
        conflict_keywords = [
            '<<', 'P.S.', 'P.P.S', 'Há algo mais', 'espaços reservados',
            'colchetes', 'aspas duplas', 'letra', 'palavras', 'criança', 'frases',
        ]
        # Start from a variation-dependent offset to pick different constraints
        for j in range(len(safe_constraints)):
            idx = (variation_idx + j) % len(safe_constraints)
            constraint = safe_constraints[idx]
            conflict = False
            for keyword in conflict_keywords:
                if keyword.lower() in prompt.lower() and keyword.lower() in constraint.lower():
                    conflict = True
                    break
            if not conflict:
                result = prompt + constraint
                break
    
    return result

def validate_variation(original, variation, var_num):
    """Log warnings if a variation seems problematic."""
    issues = []
    
    if variation == original:
        issues.append(f"  WARNING: Variation {var_num} is identical to original")
    
    if len(variation) < 10:
        issues.append(f"  WARNING: Variation {var_num} is suspiciously short")
    
    # Check that JSON structure is preserved
    if '"' in variation:
        try:
            json.dumps({"prompt": variation}, ensure_ascii=False)
        except Exception:
            issues.append(f"  WARNING: Variation {var_num} may have JSON encoding issues")
    
    return issues


def main(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    num_variations = args.num_variations
    
    # Read original prompts
    original_prompts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                original_prompts.append(data['prompt'])
    
    print(f"Read {len(original_prompts)} original prompts")
    print(f"Generating {num_variations} variation(s) per prompt...")
    
    # Generate variations
    total_warnings = 0
    identical_count = 0
    generated_prompts = []
    
    dedup_constraints = [
        ' Certifique-se de que a resposta tenha no máximo 20 frases.',
        ' Inclua pelo menos 2 exemplos concretos na sua resposta.',
        ' Não inclua a letra "z" em nenhum lugar da sua resposta.',
        ' Sua resposta deve conter entre 150 e 250 palavras.',
        ' Escreva sua resposta como se estivesse explicando para uma criança de 10 anos.',
        ' Envolva toda a resposta entre aspas duplas.',
        ' Adicione uma nota pós-escrita começando com "P.P.S" no final da sua resposta.',
        ' Sua resposta deve conter pelo menos 3 parágrafos.',
        ' Inclua um título entre duplas setas angulares, como <<título>>.',
        ' Termine sua resposta com a frase exata: "Há algo mais em que eu possa ajudar?"',
    ]
    
    for i, prompt in enumerate(original_prompts):
        variations_for_prompt = []
        
        for v_idx in range(num_variations):
            # Reset random seed per (prompt, variation) pair for reproducibility
            random.seed(42 + i * num_variations + v_idx)
            
            variation = create_variation(prompt, v_idx, num_variations)
            
            # Ensure this variation is unique among all variations for this prompt
            attempts = 0
            while variation in variations_for_prompt and attempts < len(dedup_constraints):
                candidate = variation + dedup_constraints[attempts % len(dedup_constraints)]
                if candidate not in variations_for_prompt:
                    variation = candidate
                    break
                attempts += 1
            
            # Validate
            issues = validate_variation(prompt, variation, v_idx + 1)
            if variation == prompt:
                identical_count += 1
            if issues:
                total_warnings += len(issues)
                if any('identical' in issue for issue in issues):
                    print(f"Prompt {i+1}:")
                    print(f"  Original: {prompt[:80]}...")
                    for issue in issues:
                        print(issue)
            
            variations_for_prompt.append(variation)
        
        generated_prompts.extend(variations_for_prompt)
    
    total_generated = len(generated_prompts)
    print(f"\nGenerated {total_generated} variations from {len(original_prompts)} prompts")
    
    # Combine originals + generated and deduplicate
    all_prompts = original_prompts + generated_prompts
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    
    num_duplicates = len(all_prompts) - len(unique_prompts)
    print(f"Combined {len(original_prompts)} originals + {total_generated} variations = {len(all_prompts)} total")
    print(f"Removed {num_duplicates} duplicate(s)")
    print(f"Final dataset size: {len(unique_prompts)}")
    
    # Write deduplicated output
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in unique_prompts:
            f.write(json.dumps({'prompt': p}, ensure_ascii=False) + '\n')
    
    print(f"Output written to: {output_path}")
    print(f"Identical to original: {identical_count}/{total_generated}")
    print(f"Total warnings: {total_warnings}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--num_variations', type=int, default=2,
                        help='Number of variations to generate per prompt (default: 2)')
    args = parser.parse_args()
    main(args)
