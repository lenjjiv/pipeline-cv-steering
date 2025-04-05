from load_model import load_model, run_model
from construct_prompts import (
    get_imagenet_classes, 
    get_prompts_concrete, 
    get_prompts_human_related, 
    get_prompts_style
)
from tqdm.auto import tqdm
from controller import VectorStore, register_vector_control
from collections import defaultdict
import numpy as np


def compute_steering_vectors(
    model_name='sd14', 
    mode='style', 
    concept_pos='anime', 
    concept_neg=None,
    num_denoising_steps=50,
    max_prompts=None
):
    """
    Вычисляет управляющие векторы для указанной модели и концепции
    
    Возвращает словарь векторов управления
    """
    print(f"Вычисляем управляющие векторы: модель={model_name}, режим={mode}, концепция={concept_pos}")
    
    # Загружаем модель
    pipe, device = load_model(model_name)
    
    # Формируем промпты
    if mode == 'concrete':
        prompts_pos, prompts_neg = get_prompts_concrete(
            num=max_prompts, 
            concept_pos=concept_pos, 
            concept_neg=concept_neg
        )
    elif mode == 'human-related':
        prompts_pos, prompts_neg = get_prompts_human_related(
            concept_pos=concept_pos, 
            concept_neg=concept_neg
        )
        if max_prompts is not None:
            prompts_pos, prompts_neg = prompts_pos[:max_prompts], prompts_neg[:max_prompts]

    elif mode == 'style':
        prompts_pos, prompts_neg = get_prompts_style(
            num=max_prompts, 
            concept_pos=concept_pos, 
            concept_neg=concept_neg
        )
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    # Вычисляем выходы CA для промптов
    pos_vectors = []
    neg_vectors = []
    seed = 0
    print(f"Обрабатываем {len(prompts_pos)} пар промптов...")

    iterator = tqdm(enumerate(zip(prompts_pos, prompts_neg)), total=max_prompts)
    for i, (prompt_pos, prompt_neg) in iterator:
        print(f"Промпт {i+1}/{len(prompts_pos)}: '{prompt_pos}' и '{prompt_neg}'")

        # Для положительного промпта
        controller = VectorStore(steer=False, device=device)
        register_vector_control(pipe.unet, controller)
        image = run_model(model_name, pipe, prompt_pos, seed, num_denoising_steps, device)
        pos_vectors.append(controller.vector_store)

        # Для отрицательного промпта
        controller = VectorStore(steer=False, device=device)
        register_vector_control(pipe.unet, controller)
        image = run_model(model_name, pipe, prompt_neg, seed, num_denoising_steps, device)
        neg_vectors.append(controller.vector_store)

    # Вычисляем управляющие векторы
    steering_vectors = {}

    for denoising_step in range(0, num_denoising_steps):
        steering_vectors[denoising_step] = defaultdict(list)

        for key in ['up', 'down', 'mid']:
            for layer_num in range(len(pos_vectors[0][denoising_step][key])):
                # Собираем векторы для текущего слоя
                pos_vectors_layer = [pos_vectors[i][denoising_step][key][layer_num] for i in range(len(pos_vectors))]
                pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)

                neg_vectors_layer = [neg_vectors[i][denoising_step][key][layer_num] for i in range(len(neg_vectors))]
                neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)

                # Вычисляем и нормализуем управляющий вектор
                steering_vector = pos_vectors_avg - neg_vectors_avg
                steering_vector = steering_vector / np.linalg.norm(steering_vector)

                steering_vectors[denoising_step][key].append(steering_vector)
    
    print("Управляющие векторы успешно вычислены!")
    return steering_vectors