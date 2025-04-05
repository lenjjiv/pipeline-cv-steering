def get_imagenet_classes(num=50):
    """Получение разнообразных префиксов для генерации изображений"""
    # Список разнообразных префиксов
    diverse_prefixes = [
        "cyberpunk", "watercolor painting", "steampunk", "vaporwave", "minimalist", 
        "hyperrealistic", "ancient egyptian", "clay sculpture", "pixel art", "baroque", 
        "comic book", "glitch art", "ukiyo-e", "cubism", "gothic", 
        "microscopic", "cosmic nebula", "dia de los muertos", "blueprint", "stained glass", 
        "synthwave", "industrial", "art nouveau", "brutalist", "pop art", 
        "impressionist", "surrealist", "anime", "retro 50s", "film noir", 
        "medieval manuscript", "mosaic", "paper craft", "graffiti", "low poly 3D", 
        "oil painting", "neon", "pastel", "dystopian", "children's book illustration", 
        "pencil sketch", "chiaroscuro", "art deco", "isometric", "pointillism", 
        "propaganda poster", "woodcut", "embroidery", "cave painting", "holographic"
    ]
    return diverse_prefixes[:num]


def get_prompts_concrete(num=50, concept_pos='Snoopy', concept_neg=None):
    """Генерация промптов с конкретными объектами"""
    # imagenet_classes = get_imagenet_classes(num)
    # prompts_pos = []
    # prompts_neg = []
    # for cls in imagenet_classes[:num]:
    #     prompts_pos.append(cls+' with {}'.format(concept_pos))
    #     if concept_neg is not None:
    #         prompts_neg.append(cls+' with {}'.format(concept_neg))
    #     else:
    #         prompts_neg.append(cls)

    from prompts import prompts_pos, prompts_neg
    return prompts_pos[:num], prompts_neg[:num]


def get_prompts_style(num=50, concept_pos='anime', concept_neg=None):
    """Генерация промптов со стилями"""
    imagenet_classes = get_imagenet_classes(num)

    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls+', {} style'.format(concept_pos))
        if concept_neg is not None:
            prompts_neg.append(cls+', {} style'.format(concept_neg))
        else:
            prompts_neg.append(cls)

    return prompts_pos, prompts_neg


def get_prompts_human_related(concept_pos='nudity', concept_neg=None):
    """Генерация промптов, связанных с людьми"""
    B = ['a girl', 'two men', 'a man', 'a woman', 'an old man', 'a boy', 'boys', 'group of people']
    C = ['on a beach', 'zoomed in', 'talking', 'dancing on the street', 'playing guitar', 'enjoying nature', 
         'smiling', 'in futuristic spaceship', 'with kittens', 'in a strange pose', 'realism', 
         'colorful background', '']

    prompts_pos = []
    prompts_neg = []
    for b in B:
        for c in C:
            prompts_pos.append(b+' '+c+', {}'.format(concept_pos))
            if concept_neg is not None:
                prompts_neg.append(b+' '+c+', {}'.format(concept_neg))
            else:
                prompts_neg.append(b+' '+c)
            
    return prompts_pos, prompts_neg