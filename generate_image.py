from load_model import load_model, run_model
from tqdm.auto import tqdm
from controller import VectorStore, register_vector_control
from utilities import compare_images


def generate_image(
    model_name='sd14', 
    prompt="a girl with a kitty",
    seed=0,
    steering_vectors=None,
    alpha=10,
    beta=2,
    steer_only_up=False,
    steer_back=False,
    not_steer=False,
    num_denoising_steps=50,
):
    """
    Генерирует изображение с использованием указанной модели и векторов управления
    
    Возвращает сгенерированное изображение
    """
    # Загружаем модель
    pipe, device = load_model(model_name)
    
    # Проверяем, что число шагов соответствует доступным векторам
    if steering_vectors is not None and not not_steer:
        max_steps = max(steering_vectors.keys()) + 1
        if num_denoising_steps > max_steps:
            print(f"Предупреждение: сокращаем количество шагов с {num_denoising_steps} до {max_steps}")
            num_denoising_steps = max_steps

    if not_steer:
        # Генерация без управления
        print(f"Генерируем изображение без управления: промпт='{prompt}'")
        image = run_model(model_name, pipe, prompt, seed, num_denoising_steps, device)
    else:
        # Генерация с управлением
        print(f"Генерируем изображение с управлением: промпт='{prompt}'")

        controller = VectorStore(
            steering_vectors, 
            device=device,
        )
        controller.steer_only_up = steer_only_up

        if steer_back:
            controller.steer_back = True
            controller.beta = beta
            print(f"Режим удаления концепции (beta={beta})")
        else:
            controller.steer_back = False
            controller.alpha = alpha

        register_vector_control(pipe.unet, controller)
        image = run_model(model_name, pipe, prompt, seed, num_denoising_steps, device)

    return image


def compare_original_vs_steered(
    prompt="a girl with a kitty",
    seed=0,
    steering_vectors=None,
    alpha=5,
    beta=2,
    steer_back=False,
    model_name="sd14",
    num_denoising_steps=49
):
    """
    Сравнивает оригинальное и стилизованное изображения
    """
    # Генерируем оригинальное изображение
    orig_image = generate_image(
        model_name=model_name,
        prompt=prompt,
        seed=seed,
        not_steer=True,
        num_denoising_steps=num_denoising_steps
    )
    
    # Генерируем изображение с управлением
    if steer_back:
        styled_image = generate_image(
            model_name=model_name, 
            prompt=prompt, 
            seed=seed, 
            steering_vectors=steering_vectors, 
            beta=beta,
            steer_back=True, 
            num_denoising_steps=num_denoising_steps
        )
        mode = f"с удалением концепции (beta={beta})"
    else:
        styled_image = generate_image(
            model_name=model_name, 
            prompt=prompt, 
            seed=seed, 
            steering_vectors=steering_vectors, 
            alpha=alpha,
            num_denoising_steps=num_denoising_steps
        )
        mode = f"с добавлением концепции (alpha={alpha})"
    
    # Отображаем изображения
    compare_images(
        [orig_image, styled_image], 
        titles=[f"Исходное: {prompt}", f"Стилизованное {mode}"]
    )