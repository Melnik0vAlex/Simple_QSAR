import numpy as np
import matplotlib.pyplot as plt


def plot_model_performance(
    y_train, y_pred_train, y_test, y_pred_test, model_name, column_name, metrics
):
    """
    Строит график "экспериментальное vs. предсказанное" с доверительными интервалами.

    Args:
        y_train (np.ndarray): Истинные значения обучающей выборки.
        y_pred_train (np.ndarray): Предсказания на обучающей выборке.
        y_test (np.ndarray): Истинные значения тестовой выборки.
        y_pred_test (np.ndarray): Предсказания на тестовой выборке.
        model_name (str): Название модели (отображается в заголовке).
        column_name (str): Название целевого признака (например, 'pIC50').
        metrics (dict): Словарь с метриками модели (R², RMSE, Q² и т.д.).

    Returns:
        None
    """
    residuals_train = y_pred_train - y_train
    lower_bound = np.percentile(residuals_train, 5)
    upper_bound = np.percentile(residuals_train, 95)

    ideal_line = np.linspace(min(y_train), max(y_train), 100)
    lower_interval = ideal_line + lower_bound
    upper_interval = ideal_line + upper_bound

    _, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        y_train, y_pred_train, color="blue", alpha=0.6, label="Обучающая выборка"
    )
    ax.scatter(y_test, y_pred_test, color="green", alpha=0.8, label="Тестовая выборка")

    ax.plot(
        ideal_line,
        ideal_line,
        color="red",
        linestyle="--",
        label="Идеальные предсказания",
    )
    ax.plot(
        ideal_line,
        lower_interval,
        color="orange",
        linestyle="--",
        label="Нижний доверительный интервал (90%)",
    )
    ax.plot(
        ideal_line,
        upper_interval,
        color="purple",
        linestyle="--",
        label="Верхний доверительный интервал (90%)",
    )

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel(f"Экспериментальные значения {column_name}", fontsize=12)
    ax.set_ylabel(f"Предсказанные значения {column_name}", fontsize=12)

    ax.set_title(
        f"Экспериментальные и предсказанные значения: {column_name}\nМодель: {model_name}",
        fontsize=14,
        pad=20,
    )

    # Метрики (синим — train, зелёным — test)
    ax.text(
        0.02,
        0.98,
        f"Обучающая выборка:\nR² = {metrics['R^2']:.4f}\nRMSE = {metrics['RMSE']:.4f}",
        fontsize=12,
        color="blue",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax.text(
        0.02,
        0.70,
        f"Тестовая выборка:\nPRESS = {metrics['PRESS']:.4f}\nPRMSE = {metrics['PRMSE']:.4f}\nQ² = {metrics['Q^2']:.4f}",
        fontsize=12,
        color="green",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.show()
