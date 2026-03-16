"""
Base prompt templates.

Each template defines a task type with variable slots.
Instruction constraints (modifiers) are applied separately by the generator,
guaranteeing that prompt text always matches kwargs values.

Template structure:
    id:      Unique identifier for the task type.
    prompts: List of prompt format strings with {slot} placeholders.
    slots:   Dict mapping slot names to lists of possible values.
"""

TEMPLATES = [
    #  Blog / Article 
    {
        "id": "blog_topic",
        "prompts": [
            "Escreva uma postagem de blog sobre {topic}.",
            "Crie um artigo de blog sobre {topic}.",
            "Redija uma publicação para blog abordando {topic}.",
        ],
        "slots": {
            "topic": [
                "os benefícios de dormir em uma rede",
                "como manter uma alimentação saudável",
                "dicas para aprender um novo idioma",
                "a importância da leitura diária",
                "como organizar uma viagem econômica",
                "o impacto das redes sociais na sociedade",
                "a história do café no Brasil",
                "como começar a meditar",
                "os desafios do trabalho remoto",
                "os benefícios da prática de exercícios físicos",
            ],
        },
    },
    #  Email (professional) 
    {
        "id": "email_professional",
        "prompts": [
            "Escreva um e-mail para {recipient} informando que {subject}.",
            "Redija um e-mail profissional para {recipient} sobre o fato de que {subject}.",
            "Elabore um e-mail formal para {recipient} comunicando que {subject}.",
        ],
        "slots": {
            "recipient": [
                "meu chefe",
                "minha gerente",
                "o departamento de recursos humanos",
                "meu orientador",
                "a equipe do projeto",
            ],
            "subject": [
                "estou pedindo demissão",
                "preciso de férias na próxima semana",
                "gostaria de solicitar um aumento salarial",
                "não poderei comparecer à reunião de amanhã",
                "concluí o projeto antes do prazo estipulado",
                "haverá uma reestruturação na equipe",
            ],
        },
    },
    #  Email (invitation) 
    {
        "id": "email_invitation",
        "prompts": [
            "Escreva um modelo de e-mail que convida {audience} para {event}.",
            "Redija um e-mail convidando {audience} para participar de {event}.",
            "Elabore um convite por e-mail para {audience} sobre {event}.",
        ],
        "slots": {
            "audience": [
                "um grupo de participantes",
                "os membros da equipe",
                "os funcionários da empresa",
                "os alunos da turma",
                "os voluntários do projeto",
            ],
            "event": [
                "uma reunião estratégica",
                "um workshop de inovação",
                "uma palestra sobre sustentabilidade",
                "um treinamento técnico",
                "uma confraternização de fim de ano",
            ],
        },
    },
    #  Letter (informal) 
    {
        "id": "letter_informal",
        "prompts": [
            "Escreva uma carta para {recipient} {purpose}.",
            "Redija uma carta informal para {recipient} {purpose}.",
            "Elabore uma carta destinada a {recipient} {purpose}.",
        ],
        "slots": {
            "recipient": [
                "um amigo",
                "um parente distante",
                "um colega de escola",
                "seu vizinho",
                "um antigo professor",
            ],
            "purpose": [
                "pedindo que ele vá votar",
                "convidando-o para um jantar",
                "agradecendo um favor recente",
                "pedindo desculpas por um mal-entendido",
                "contando sobre sua viagem recente",
                "pedindo conselhos sobre uma decisão importante",
            ],
        },
    },
    #  Resume / CV 
    {
        "id": "resume",
        "prompts": [
            "Escreva um currículo para {person}.",
            "Elabore um currículo profissional para {person}.",
            "Crie um currículo detalhado para {person}.",
        ],
        "slots": {
            "person": [
                "um recém-formado no ensino médio que está buscando seu primeiro emprego",
                "um engenheiro de software com 5 anos de experiência",
                "um professor universitário de história",
                "um chef de cozinha que busca oportunidades internacionais",
                "uma enfermeira com experiência em UTI",
                "o palhaço profissional Bozo",
                "Carlos Mendes, um designer gráfico freelancer",
                "Fernanda Costa, uma advogada especializada em direito ambiental",
            ],
        },
    },
    #  Story / Narrative 
    {
        "id": "story",
        "prompts": [
            "Escreva uma história sobre {plot}.",
            "Crie uma narrativa sobre {plot}.",
            "Redija um conto sobre {plot}.",
        ],
        "slots": {
            "plot": [
                "um homem que acorda um dia e percebe que ele estava dentro de um jogo eletrônico",
                "uma cientista que descobre um portal para outra dimensão",
                "um gato que aprende a falar e decide se candidatar a prefeito",
                "uma criança que encontra um mapa do tesouro no sótão",
                "um robô que desenvolve sentimentos e questiona sua existência",
                "uma astronauta que encontra vida em Marte",
                "um músico que só consegue tocar quando está triste",
                "uma bibliotecária que descobre que os livros ganham vida à noite",
            ],
        },
    },
    #  Poem 
    {
        "id": "poem",
        "prompts": [
            "Escreva um poema sobre {theme}.",
            "Componha um poema sobre {theme}.",
            "Crie um poema inspirado em {theme}.",
        ],
        "slots": {
            "theme": [
                "como estou perdendo minhas aulas",
                "a saudade de um amor distante",
                "o pôr do sol visto da praia",
                "a passagem do tempo",
                "a beleza das pequenas coisas",
                "uma noite estrelada no campo",
                "a força da amizade",
                "a chegada da primavera",
            ],
        },
    },
    #  Song Lyrics 
    {
        "id": "song_lyrics",
        "prompts": [
            "Escreva a letra de uma música {style} sobre {theme}.",
            "Crie a letra de uma canção {style} sobre {theme}.",
            "Componha uma letra musical {style} sobre {theme}.",
        ],
        "slots": {
            "style": [
                "de rock",
                "de samba",
                "de bossa nova",
                "de MPB",
                "de rap",
                "de sertanejo",
                "de forró",
                "pop",
            ],
            "theme": [
                "liberdade e independência",
                "um amor de verão na praia",
                "superar obstáculos na vida",
                "a cidade grande e seus contrastes",
                "uma viagem sem destino",
                "memórias de infância",
            ],
        },
    },
    #  Dialogue 
    {
        "id": "dialogue",
        "prompts": [
            "Escreva um diálogo entre {characters} sobre {situation}.",
            "Crie uma conversa entre {characters} discutindo sobre {situation}.",
            "Elabore um diálogo entre {characters} a respeito de {situation}.",
        ],
        "slots": {
            "characters": [
                "duas pessoas, uma vestida com um vestido de baile e a outra com roupas esportivas",
                "um professor e um aluno",
                "dois vizinhos que nunca se falaram",
                "um médico e um paciente",
                "uma mãe e um filho adolescente",
                "dois astronautas na Estação Espacial Internacional",
            ],
            "situation": [
                "irem juntos a um evento noturno",
                "a melhor forma de resolver um conflito",
                "planos para o final de semana",
                "o significado da felicidade",
                "uma descoberta científica surpreendente",
                "a escolha de um destino de viagem",
            ],
        },
    },
    #  Advertisement / Marketing 
    {
        "id": "advertisement",
        "prompts": [
            "Crie uma publicidade para {product}.",
            "Elabore um anúncio publicitário para {product}.",
            "Escreva um texto de marketing para {product}.",
        ],
        "slots": {
            "product": [
                "uma fralda projetada para ser mais confortável para bebês",
                "um novo aplicativo de meditação",
                "um carro elétrico acessível",
                "um serviço de delivery de comida caseira",
                "uma linha de cosméticos veganos",
                "um novo tipo de sorvete chamado 'Sorvete Solar'",
                "um tênis de corrida feito de material reciclado",
                "uma escola de idiomas online",
            ],
        },
    },
    #  Review / Critique 
    {
        "id": "review",
        "prompts": [
            "Escreva uma crítica sobre {subject}.",
            "Elabore uma resenha sobre {subject}.",
            "Redija uma análise crítica sobre {subject}.",
        ],
        "slots": {
            "subject": [
                "o filme 'Cidade de Deus'",
                "o livro 'Dom Casmurro' de Machado de Assis",
                "o álbum mais recente de uma banda indie brasileira",
                "um restaurante japonês em São Paulo",
                "a série documental sobre a Amazônia",
                "o impacto da inteligência artificial na educação",
                "o espetáculo teatral 'O Auto da Compadecida'",
            ],
        },
    },
    #  Travel Itinerary 
    {
        "id": "travel_itinerary",
        "prompts": [
            "Escreva um roteiro de viagem para {destination}.",
            "Crie um guia de viagem para {destination}.",
            "Elabore um itinerário turístico para {destination}.",
        ],
        "slots": {
            "destination": [
                "o Japão",
                "Portugal",
                "a Argentina",
                "o Chile",
                "a Tailândia",
                "a Itália",
                "o Peru",
                "a Grécia",
                "a Colômbia",
                "o México",
            ],
        },
    },
    #  Explanation 
    {
        "id": "explanation",
        "prompts": [
            "Explique {topic}.",
            "Descreva em detalhes {topic}.",
            "Forneça uma explicação clara sobre {topic}.",
        ],
        "slots": {
            "topic": [
                "por que é importante comer alimentos saudáveis para curar o corpo",
                "como funciona a fotossíntese",
                "por que o céu é azul",
                "como surgiu a internet",
                "o que é inteligência artificial e como ela funciona",
                "a importância da reciclagem para o meio ambiente",
                "como os terremotos acontecem",
                "por que os idiomas mudam ao longo do tempo",
            ],
        },
    },
    #  Summary 
    {
        "id": "summary",
        "prompts": [
            "Escreva um resumo sobre {topic}.",
            "Faça um resumo detalhado sobre {topic}.",
            "Redija uma síntese sobre {topic}.",
        ],
        "slots": {
            "topic": [
                "o descobrimento do Brasil",
                "a Revolução Francesa",
                "a teoria da evolução de Darwin",
                "a história da computação",
                "a conquista do espaço",
                "o aquecimento global",
                "a história do futebol no Brasil",
                "a Revolução Industrial",
            ],
        },
    },
    #  Speech 
    {
        "id": "speech",
        "prompts": [
            "Escreva um discurso {occasion}.",
            "Elabore um discurso para {occasion}.",
            "Redija um discurso adequado para {occasion}.",
        ],
        "slots": {
            "occasion": [
                "de formatura para uma turma de engenharia",
                "de abertura de um evento de tecnologia",
                "de agradecimento ao receber um prêmio literário",
                "motivacional para uma equipe de vendas",
                "de encerramento de ano em uma escola",
                "de posse de um novo presidente de associação comunitária",
            ],
        },
    },
    #  Social Media 
    {
        "id": "social_media",
        "prompts": [
            "Crie uma postagem para {platform} sobre {topic}.",
            "Escreva um texto para {platform} abordando {topic}.",
            "Elabore uma publicação para {platform} sobre {topic}.",
        ],
        "slots": {
            "platform": [
                "o Instagram",
                "o Twitter",
                "o LinkedIn",
                "o Facebook",
                "o TikTok",
            ],
            "topic": [
                "dicas de produtividade",
                "uma receita rápida e saudável",
                "a importância da saúde mental",
                "uma conquista profissional recente",
                "um livro que mudou sua perspectiva",
                "motivação para a segunda-feira",
            ],
        },
    },
    #  Pitch 
    {
        "id": "pitch",
        "prompts": [
            "Escreva uma apresentação de pitch para {product}.",
            "Crie um pitch de vendas para {product}.",
            "Elabore uma proposta comercial para {product}.",
        ],
        "slots": {
            "product": [
                "um novo tipo de sorvete que é suave no estômago",
                "um aplicativo que conecta donos de pets a passeadores",
                "uma startup de energia solar residencial",
                "um serviço de tutoria online com inteligência artificial",
                "uma plataforma de crowdfunding para projetos sociais",
                "um sistema de irrigação inteligente para pequenos agricultores",
            ],
        },
    },
    #  Tutorial / Instructions 
    {
        "id": "tutorial",
        "prompts": [
            "Escreva um tutorial sobre como {task}.",
            "Crie um guia passo a passo sobre como {task}.",
            "Elabore instruções detalhadas sobre como {task}.",
        ],
        "slots": {
            "task": [
                "preparar um café perfeito",
                "montar um currículo atrativo",
                "organizar um evento beneficente",
                "instalar um sistema operacional Linux",
                "cultivar uma horta em apartamento",
                "escrever um bom relatório técnico",
                "planejar uma festa surpresa",
                "fazer uma apresentação profissional em público",
            ],
        },
    },
    #  List 
    {
        "id": "list",
        "prompts": [
            "Forneça uma lista de {items}.",
            "Liste {items}.",
            "Enumere {items}.",
        ],
        "slots": {
            "items": [
                "5 destinos turísticos imperdíveis no Brasil",
                "os 10 melhores livros de literatura brasileira",
                "7 hábitos saudáveis para o dia a dia",
                "5 mães famosas na história",
                "8 invenções que mudaram o mundo",
                "6 receitas fáceis para iniciantes na cozinha",
                "5 dicas para economizar dinheiro no dia a dia",
            ],
        },
    },
    #  Quiz 
    {
        "id": "quiz",
        "prompts": [
            "Crie um quiz {quiz_type} sobre {topic}.",
            "Elabore um questionário {quiz_type} sobre {topic}.",
            "Monte um quiz {quiz_type} a respeito de {topic}.",
        ],
        "slots": {
            "quiz_type": [
                "lógico",
                "de conhecimentos gerais",
                "educativo",
                "divertido",
                "desafiador",
            ],
            "topic": [
                "a história do Brasil",
                "ciência e tecnologia",
                "geografia mundial",
                "literatura clássica",
                "cultura pop dos anos 90",
                "esportes olímpicos",
                "curiosidades sobre animais",
            ],
        },
    },
    #  Recipe 
    {
        "id": "recipe",
        "prompts": [
            "Escreva uma receita de {dish}.",
            "Elabore uma receita detalhada de {dish}.",
            "Crie uma receita de {dish} com instruções passo a passo.",
        ],
        "slots": {
            "dish": [
                "bolo de chocolate",
                "feijoada tradicional",
                "pão de queijo mineiro",
                "moqueca de peixe",
                "brigadeiro gourmet",
                "tapioca recheada",
                "coxinha cremosa",
                "açaí na tigela",
            ],
        },
    },
    #  Rewrite (formal) 
    {
        "id": "rewrite",
        "prompts": [
            "Reescreva a seguinte declaração para que soe mais formal: \"{statement}\"",
            "Reformule o seguinte texto em um tom mais formal: \"{statement}\"",
            "Torne a seguinte frase mais formal e profissional: \"{statement}\"",
        ],
        "slots": {
            "statement": [
                "A gente precisa resolver isso logo antes que fique pior.",
                "Esse projeto tá muito legal mas precisa de uns ajustes.",
                "Acho que deveríamos mudar nossa estratégia pra conseguir melhores resultados.",
                "O time fez um trabalho massa e merece reconhecimento.",
                "Precisamos conversar sobre o que aconteceu na reunião.",
            ],
        },
    },
    #  Sentiment Analysis 
    {
        "id": "sentiment",
        "prompts": [
            "Dada a frase \"{phrase}\", o sentimento é positivo ou negativo?",
            "Analise o sentimento da seguinte frase: \"{phrase}\"",
            "Classifique como positivo ou negativo o sentimento expresso em: \"{phrase}\"",
        ],
        "slots": {
            "phrase": [
                "Não está claro quanto desse dinheiro realmente está sendo gasto com crianças",
                "O resultado superou todas as nossas expectativas",
                "Infelizmente não conseguimos atingir a meta estabelecida",
                "A equipe demonstrou grande empenho e dedicação",
                "Os números mostram uma queda preocupante nas vendas",
                "Estamos muito satisfeitos com o progresso alcançado",
            ],
        },
    },
    #  Joke 
    {
        "id": "joke",
        "prompts": [
            "Escreva uma piada sobre {topic}.",
            "Crie uma piada engraçada sobre {topic}.",
            "Conte uma piada envolvendo {topic}.",
        ],
        "slots": {
            "topic": [
                "foguetes",
                "programadores",
                "gatos e cachorros",
                "matemática",
                "viagens de avião",
                "o tempo",
                "comida brasileira",
                "futebol",
            ],
        },
    },
    #  Haiku 
    {
        "id": "haiku",
        "prompts": [
            "Escreva um haicai sobre {theme}.",
            "Componha um haiku sobre {theme}.",
            "Crie um haicai inspirado em {theme}.",
        ],
        "slots": {
            "theme": [
                "mães",
                "a natureza",
                "o mar",
                "o outono",
                "a lua",
                "o silêncio",
                "a chuva",
                "o amanhecer",
            ],
        },
    },
    #  Report 
    {
        "id": "report",
        "prompts": [
            "Escreva um relatório sobre {topic}.",
            "Elabore um relatório detalhado sobre {topic}.",
            "Redija um relatório técnico sobre {topic}.",
        ],
        "slots": {
            "topic": [
                "o desempenho trimestral de vendas de uma loja online",
                "o impacto ambiental do desmatamento na Amazônia",
                "as tendências do mercado de tecnologia em 2025",
                "os resultados de uma pesquisa de satisfação de clientes",
                "a implementação de um novo sistema de gestão",
                "a qualidade da água em rios urbanos brasileiros",
            ],
        },
    },
    #  Comparison 
    {
        "id": "comparison",
        "prompts": [
            "Compare {pair}, destacando semelhanças e diferenças.",
            "Faça uma comparação detalhada entre {pair}.",
            "Analise as diferenças e semelhanças entre {pair}.",
        ],
        "slots": {
            "pair": [
                "trabalho remoto e trabalho presencial",
                "inteligência artificial e inteligência humana",
                "energia solar e energia eólica",
                "café e chá",
                "monarquia e república",
                "música clássica e música popular",
            ],
        },
    },
    #  Opinion / Essay 
    {
        "id": "opinion",
        "prompts": [
            "Escreva um texto argumentativo sobre {topic}.",
            "Elabore uma opinião fundamentada sobre {topic}.",
            "Redija um ensaio sobre {topic}.",
        ],
        "slots": {
            "topic": [
                "a obrigatoriedade do voto no Brasil",
                "o uso de uniformes em escolas",
                "a redução da jornada de trabalho",
                "a proibição de celulares em salas de aula",
                "a importância da educação financeira desde a infância",
                "os prós e contras do ensino a distância",
            ],
        },
    },
    #  Children's Story 
    {
        "id": "children_story",
        "prompts": [
            "Escreva uma história infantil sobre {plot}.",
            "Crie um conto para crianças sobre {plot}.",
            "Elabore uma história infantil envolvendo {plot}.",
        ],
        "slots": {
            "plot": [
                "um coelhinho que queria voar",
                "uma árvore mágica que concedia desejos",
                "um dragão que tinha medo de fogo",
                "uma estrela que caiu do céu",
                "um robôzinho que procurava um amigo",
                "uma princesa que virou aventureira",
            ],
        },
    },
    #  Person Description 
    {
        "id": "person_description",
        "prompts": [
            "Descreva {person} em detalhes, incluindo aparência e personalidade.",
            "Crie uma descrição detalhada de {person}.",
            "Escreva uma descrição completa de {person}.",
        ],
        "slots": {
            "person": [
                "um detetive aposentado que mora em uma cidade pequena",
                "uma artista de rua em São Paulo",
                "um pescador idoso no litoral nordestino",
                "uma cientista brilhante que é introvertida",
                "um chef de cozinha apaixonado por ingredientes regionais",
                "uma professora rural que transforma vidas",
            ],
        },
    },
]
