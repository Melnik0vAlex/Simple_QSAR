import numpy as np
import joblib

from scipy.stats import spearmanr
from sklearn.linear_model import (
    LinearRegression,
    ElasticNet,
    HuberRegressor,
    Ridge,
    Lasso,
    BayesianRidge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


class RegressionModel:
    """
    Класс для построения, обучения и оценки моделей регрессии в QSAR-задачах.
    Поддерживает выбор модели, кросс-валидацию, визуализацию, подбор гиперпараметров.
    """

    def __init__(self, models=None):
        """
        Инициализация модели с заданным набором регрессоров.

        Args:
            models (dict, optional): Пользовательский словарь {имя: модель}.
        """
        if models is None:
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(
                    random_state=42,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.6,
                ),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "ElasticNet": ElasticNet(),
                "Huber": HuberRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso(alpha=0.01),
                "BayesianRidge": BayesianRidge(),
            }

        self.models = models
        self.current_model_name = None
        self.current_model = None
        self.X_train, self.X_test = None, None  # pylint: disable=invalid-name
        self.y_train, self.y_test = None, None
        self.y_pred_train, self.y_pred_test = None, None

    def add_model(self, model_name, model_instance):
        """Добавляет новую модель в список."""
        self.models[model_name] = model_instance

    def select_model(self, model_name):
        """Устанавливает текущую модель по имени."""
        if model_name in self.models:
            self.current_model_name = model_name
            self.current_model = self.models[model_name]
        else:
            raise ValueError(f"Модель '{model_name}' не найдена.")

    def prepare_data(
        self, X, y, test_size=0.2, random_state=42
    ):  # pylint: disable=invalid-name,redefined-outer-name
        """Разделяет данные на обучающую и тестовую выборки."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def set_data_from_split(
        self, X_train, X_test, y_train, y_test
    ):  # pylint: disable=invalid-name
        """Устанавливает данные вручную."""
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def train(self):
        """Обучает текущую модель."""
        if self.current_model is None:
            raise ValueError("Модель не выбрана.")
        self.current_model.fit(self.X_train, self.y_train)
        self.y_pred_train = self.current_model.predict(self.X_train)
        self.y_pred_test = self.current_model.predict(self.X_test)

    def predict(self, X):  # pylint: disable=invalid-name,redefined-outer-name
        """Выполняет предсказание для новых данных."""
        if self.current_model is None:
            raise ValueError("Модель не выбрана.")
        return self.current_model.predict(X)

    def save_model(self, path):
        """Сохраняет модель в файл."""
        if self.current_model is None:
            raise ValueError("Нет модели для сохранения.")
        joblib.dump(self.current_model, path)

    def load_model(self, path):
        """Загружает модель из файла."""
        self.current_model = joblib.load(path)

    def evaluate_descriptive_metrics(self):
        """Вычисляет описательные и прогностические метрики модели."""
        if self.y_train is None or self.y_pred_train is None:
            raise ValueError("Модель не обучена.")

        rss = np.sum((self.y_pred_train - self.y_train) ** 2)
        ss = np.sum((self.y_train - np.mean(self.y_train)) ** 2)
        r_squared = (ss - rss) / ss
        rmse = np.sqrt(rss / len(self.y_train))
        r = np.corrcoef(self.y_train, self.y_pred_train)[0, 1]
        rho, _ = spearmanr(self.y_train, self.y_pred_train)

        press = np.sum((self.y_pred_test - self.y_test) ** 2)
        prmse = np.sqrt(press / len(self.y_test))
        q_squared = (ss - press) / ss if ss > 0 else float("-inf")

        return {
            "RSS": rss,
            "SS": ss,
            "R^2": r_squared,
            "RMSE": rmse,
            "Pearson r": r,
            "Spearman rho": rho,
            "PRESS": press,
            "PRMSE": prmse,
            "Q^2": q_squared,
        }

    def cross_validate(
        self, X, y, cv=5, scoring="r2", n_jobs=-1
    ):  # pylint: disable=invalid-name,redefined-outer-name
        """Кросс-валидация модели с выводом средней и std метрик."""
        if self.current_model is None:
            raise ValueError("Модель не выбрана.")
        scores = cross_val_score(
            self.current_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        return {
            "Mean": scores.mean(),
            "Std": scores.std(),
            "Scores": scores.tolist(),
        }

    def grid_search(self, model_name, param_grid, cv=5, n_jobs=-1):
        """Поиск оптимальных гиперпараметров с использованием GridSearchCV."""
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не найдена.")
        grid_search = GridSearchCV(
            self.models[model_name], param_grid, cv=cv, n_jobs=n_jobs, verbose=2
        )
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_, grid_search.best_score_


# === ШАБЛОН ИСПОЛЬЗОВАНИЯ ===
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    # Загружаем примерные данные
    data = load_diabetes()
    X = data.data
    y = data.target

    # Создаём и настраиваем модель
    model = RegressionModel()
    model.select_model("RandomForest")
    model.prepare_data(X, y)
    model.train()

    # Метрики
    print(model.evaluate_descriptive_metrics())

    # Кросс-валидация
    print(model.cross_validate(X, y))
