# -*- coding: utf-8 -*-
"""
Интеллектуальный сервис рекомендации товаров и услуг.
Графический интерфейс пользователя (tkinter).

Функции (по требованиям программы государственной итоговой аттестации):
- загрузка данных для обучения из локальных файлов (MovieLens 100K);
- обучение нейросетевых моделей с сохранением в файл;
- загрузка обученной модели из файла;
- применение модели: формирование персональных рекомендаций;
- оценка качества модели на тестовой выборке.

Запуск: python gui.py
"""

import os
import sys
import threading
import queue

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch

from config import RATING_MIN, RATING_MAX
from data_processing import RecommenderDataProcessor
from neural_models import GMF, MLP_Recommender, NeuralCollaborativeFiltering, \
    WideAndDeep, NeuralModelTrainer

MODEL_CLASSES = {
    "GMF": lambda nu, ni: GMF(nu, ni, embed_dim=64),
    "MLP": lambda nu, ni: MLP_Recommender(nu, ni, 64, [128, 64, 32]),
    "NCF": lambda nu, ni: NeuralCollaborativeFiltering(nu, ni, 32, 32, [64, 32, 16]),
    "Wide & Deep": lambda nu, ni: WideAndDeep(nu, ni, 32, [64, 32]),
}


class RecommenderApp:
    """Главное окно интеллектуального сервиса рекомендаций."""

    def __init__(self, root):
        self.root = root
        self.root.title("Интеллектуальный сервис рекомендаций — Нефедов А.Г., МУИВ")
        self.root.geometry("980x640")

        self.processor = None
        self.train_data = None
        self.test_data = None
        self.matrix = None
        self.n_users = 0
        self.n_items = 0
        self.model = None
        self.model_name = None
        self.item_names = {}
        self.log_queue = queue.Queue()

        self._build_ui()
        self.root.after(200, self._poll_log)

    # ------------------------------------------------------------- интерфейс
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=(14, 6), font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self._show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)
        self.root.config(menu=menubar)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_data = ttk.Frame(self.nb)
        self.tab_train = ttk.Frame(self.nb)
        self.tab_model = ttk.Frame(self.nb)
        self.tab_rec = ttk.Frame(self.nb)
        self.tab_eval = ttk.Frame(self.nb)

        self.nb.add(self.tab_data, text="1. Данные")
        self.nb.add(self.tab_train, text="2. Обучение")
        self.nb.add(self.tab_model, text="3. Модель")
        self.nb.add(self.tab_rec, text="4. Рекомендации")
        self.nb.add(self.tab_eval, text="5. Оценка качества")

        self._build_tab_data()
        self._build_tab_train()
        self._build_tab_model()
        self._build_tab_rec()
        self._build_tab_eval()

        self.status = ttk.Label(self.root, text="Готов к работе. Начните с загрузки данных.",
                                style="Status.TLabel", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

        # горячие клавиши
        self.root.bind("<F1>", lambda e: self._show_about())
        self.root.bind("<F5>", lambda e: self.load_movielens())
        self.root.bind("<Control-s>", lambda e: self.export_recommendations())

    def _build_tab_data(self):
        f = self.tab_data
        ttk.Label(f, text="Загрузка и анализ данных", style="Header.TLabel").pack(
            anchor="w", padx=16, pady=(16, 8))
        row = ttk.Frame(f); row.pack(anchor="w", padx=16, pady=4)
        ttk.Button(row, text="Загрузить MovieLens 100K",
                   command=self.load_movielens).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Открыть CSV с оценками…",
                   command=self.load_csv).pack(side="left")

        self.data_info = tk.Text(f, height=18, width=100, font=("Consolas", 10),
                                 state="disabled", bg="#f8f8f8")
        self.data_info.pack(fill="both", expand=True, padx=16, pady=12)

    def _build_tab_train(self):
        f = self.tab_train
        ttk.Label(f, text="Обучение нейросетевой модели",
                  style="Header.TLabel").pack(anchor="w", padx=16, pady=(16, 8))

        row = ttk.Frame(f); row.pack(anchor="w", padx=16, pady=4)
        ttk.Label(row, text="Архитектура:").pack(side="left")
        self.model_var = tk.StringVar(value="Wide & Deep")
        ttk.Combobox(row, textvariable=self.model_var, state="readonly", width=14,
                     values=list(MODEL_CLASSES.keys())).pack(side="left", padx=8)
        ttk.Label(row, text="Эпох:").pack(side="left", padx=(16, 0))
        self.epochs_var = tk.IntVar(value=20)
        ttk.Spinbox(row, from_=1, to=100, width=5,
                    textvariable=self.epochs_var).pack(side="left", padx=8)
        ttk.Label(row, text="Размер батча:").pack(side="left", padx=(16, 0))
        self.batch_var = tk.IntVar(value=256)
        ttk.Spinbox(row, from_=32, to=2048, increment=32, width=6,
                    textvariable=self.batch_var).pack(side="left", padx=8)

        row2 = ttk.Frame(f); row2.pack(anchor="w", padx=16, pady=8)
        self.btn_train = ttk.Button(row2, text="Обучить модель", command=self.train_model)
        self.btn_train.pack(side="left")
        self.progress = ttk.Progressbar(row2, length=320, mode="determinate")
        self.progress.pack(side="left", padx=16)
        ttk.Button(row2, text="Очистить журнал",
                   command=self.clear_train_log).pack(side="left")

        self.train_log = tk.Text(f, height=16, width=100, font=("Consolas", 10),
                                 state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.train_log.pack(fill="both", expand=True, padx=16, pady=8)

    def _build_tab_model(self):
        f = self.tab_model
        ttk.Label(f, text="Сохранение и загрузка модели",
                  style="Header.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        row = ttk.Frame(f); row.pack(anchor="w", padx=16, pady=8)
        ttk.Button(row, text="Сохранить модель в файл…",
                   command=self.save_model).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Загрузить модель из файла…",
                   command=self.load_model).pack(side="left")

        self.model_info = tk.Text(f, height=14, width=100, font=("Consolas", 10),
                                  state="disabled", bg="#f8f8f8")
        self.model_info.pack(fill="both", expand=True, padx=16, pady=12)

    def _build_tab_rec(self):
        f = self.tab_rec
        ttk.Label(f, text="Персональные рекомендации",
                  style="Header.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        row = ttk.Frame(f); row.pack(anchor="w", padx=16, pady=4)
        ttk.Label(row, text="Идентификатор пользователя (1–943):").pack(side="left")
        self.user_var = tk.IntVar(value=42)
        ttk.Spinbox(row, from_=1, to=100000, width=8,
                    textvariable=self.user_var).pack(side="left", padx=8)
        ttk.Label(row, text="Количество рекомендаций:").pack(side="left", padx=(16, 0))
        self.topn_var = tk.IntVar(value=10)
        ttk.Spinbox(row, from_=1, to=50, width=5,
                    textvariable=self.topn_var).pack(side="left", padx=8)
        ttk.Button(row, text="Сформировать рекомендации",
                   command=self.recommend).pack(side="left", padx=16)
        ttk.Button(row, text="Сохранить в CSV…",
                   command=self.export_recommendations).pack(side="left")

        cols = ("n", "film", "year", "score")
        self.rec_tree = ttk.Treeview(f, columns=cols, show="headings", height=15)
        self.rec_tree.heading("n", text="№")
        self.rec_tree.heading("film", text="Рекомендуемый фильм")
        self.rec_tree.heading("year", text="Год")
        self.rec_tree.heading("score", text="Прогноз оценки")
        self.rec_tree.column("n", width=50, anchor="center")
        self.rec_tree.column("film", width=520)
        self.rec_tree.column("year", width=90, anchor="center")
        self.rec_tree.column("score", width=140, anchor="center")
        self.rec_tree.pack(fill="both", expand=True, padx=16, pady=12)

    def _build_tab_eval(self):
        f = self.tab_eval
        ttk.Label(f, text="Оценка качества модели на тестовой выборке",
                  style="Header.TLabel").pack(anchor="w", padx=16, pady=(16, 8))
        ttk.Button(f, text="Выполнить оценку",
                   command=self.evaluate).pack(anchor="w", padx=16, pady=8)

        self.eval_info = tk.Text(f, height=14, width=100, font=("Consolas", 11),
                                 state="disabled", bg="#f8f8f8")
        self.eval_info.pack(fill="both", expand=True, padx=16, pady=12)

    def _show_about(self):
        messagebox.showinfo(
            "О программе",
            "Интеллектуальный сервис рекомендации товаров и услуг\n\n"
            "Выпускная квалификационная работа\n"
            "Нефедов Алексей Геннадьевич\n"
            "МУИВ им. С.Ю. Витте, 09.03.03 Прикладная информатика\n\n"
            "Архитектуры: GMF, MLP, NCF, Wide & Deep (PyTorch)\n"
            "Репозиторий: github.com/ngpless/Nefedov_AG_practice")

    # ------------------------------------------------------------- утилиты
    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def clear_train_log(self):
        """Очистка журнала обучения и сброс индикатора прогресса."""
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", "end")
        self.train_log.configure(state="disabled")
        self.progress.configure(value=0)

    def _append_log(self, text):
        self.log_queue.put(("log", text))

    def _poll_log(self):
        """Обработка событий из потока обучения (только в главном потоке)."""
        while not self.log_queue.empty():
            kind, payload = self.log_queue.get_nowait()
            if kind == "log":
                self.train_log.configure(state="normal")
                self.train_log.insert("end", payload + "\n")
                self.train_log.see("end")
                self.train_log.configure(state="disabled")
            elif kind == "progress":
                self.progress.configure(value=payload)
            elif kind == "status":
                self.status.configure(text=payload)
            elif kind == "done":
                self.btn_train.configure(state="normal")
                self._show_model_info()
        self.root.after(200, self._poll_log)

    def _load_item_names(self):
        path = os.path.join("data", "raw", "ml-100k", "u.item")
        self.item_names = {}
        if os.path.exists(path):
            with open(path, encoding="latin-1") as fh:
                for line in fh:
                    parts = line.split("|")
                    self.item_names[int(parts[0])] = parts[1]

    # ------------------------------------------------------------- данные
    def load_movielens(self):
        try:
            self.status.configure(text="Загрузка MovieLens 100K…")
            self.root.update_idletasks()
            self.processor = RecommenderDataProcessor(data_dir="data")
            (self.train_data, self.test_data,
             self.matrix, info) = self.processor.process_data("movielens-100k", 0.2)
            self.n_users = len(self.processor.user_mapping)
            self.n_items = len(self.processor.item_mapping)
            self._load_item_names()
            txt = (
                "Набор данных: MovieLens 100K\n"
                f"Оценок всего:            {info.get('n_ratings', '—'):,}\n"
                f"Пользователей:           {self.n_users}\n"
                f"Объектов (фильмов):      {self.n_items}\n"
                f"Обучающая выборка:       {len(self.train_data):,} оценок\n"
                f"Тестовая выборка:        {len(self.test_data):,} оценок\n"
                f"Средняя оценка (train):  {self.train_data['rating'].mean():.3f}\n"
                f"Разреженность (train):   "
                f"{100 * (1 - len(self.train_data) / (self.n_users * self.n_items)):.2f} %\n"
                "\nДанные готовы. Перейдите на вкладку «2. Обучение»."
            ).replace(",", " ")
            self._set_text(self.data_info, txt)
            self.status.configure(text="Данные загружены.")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки данных", str(e))
            self.status.configure(text="Ошибка загрузки данных.")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="CSV с колонками user_id, item_id, rating",
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            need = {"user_id", "item_id", "rating"}
            if not need.issubset(df.columns):
                raise ValueError("CSV должен содержать колонки user_id, item_id, rating")
            self.processor = RecommenderDataProcessor(data_dir="data")
            df = self.processor.create_mappings(df)
            self.train_data, self.test_data = self.processor.split_data(df, 0.2)
            self.matrix = self.processor.create_user_item_matrix(self.train_data)
            self.n_users = len(self.processor.user_mapping)
            self.n_items = len(self.processor.item_mapping)
            self._set_text(self.data_info,
                           f"Файл: {path}\nОценок: {len(df):,}\n"
                           f"Пользователей: {self.n_users}\nОбъектов: {self.n_items}\n"
                           f"Train: {len(self.train_data):,}  Test: {len(self.test_data):,}"
                           .replace(",", " "))
            self.status.configure(text="Пользовательский CSV загружен.")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки CSV", str(e))

    # ------------------------------------------------------------- обучение
    def train_model(self):
        if self.train_data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные на вкладке «1. Данные».")
            return
        self.btn_train.configure(state="disabled")
        name = self.model_var.get()
        epochs = self.epochs_var.get()
        batch = self.batch_var.get()
        self.progress.configure(maximum=epochs, value=0)
        threading.Thread(target=self._train_worker,
                         args=(name, epochs, batch), daemon=True).start()

    def _train_worker(self, name, epochs, batch):
        try:
            self._append_log(f"=== Обучение модели {name}: эпох {epochs}, батч {batch} ===")
            trainer = NeuralModelTrainer(device="cpu")
            model = MODEL_CLASSES[name](self.n_users, self.n_items)
            self._append_log(f"Параметров модели: {model.count_parameters():,}")

            train_loader, val_loader, test_loader = trainer.create_train_val_test_loaders(
                self.train_data, self.test_data, batch)

            import torch.nn as nn
            import torch.optim as optim
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            best_val, best_state, patience = float("inf"), None, 0

            for epoch in range(epochs):
                model.train()
                losses = []
                for u, i, r in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(u, i), r)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                vlosses = []
                with torch.no_grad():
                    for u, i, r in val_loader:
                        vlosses.append(criterion(model(u, i), r).item())
                tr, vl = float(np.mean(losses)), float(np.mean(vlosses))
                self._append_log(f"Эпоха {epoch + 1:>3}/{epochs}: "
                                 f"train RMSE={tr ** 0.5:.4f}  val RMSE={vl ** 0.5:.4f}")
                self.log_queue.put(("progress", epoch + 1))
                if vl < best_val - 0.001:
                    best_val, best_state, patience = vl, model.state_dict().copy(), 0
                else:
                    patience += 1
                    if patience >= 7:
                        self._append_log(f"Ранняя остановка на эпохе {epoch + 1}.")
                        break
            if best_state:
                model.load_state_dict(best_state)
            self.model, self.model_name = model, name
            self._append_log(f"Обучение завершено. Лучший val RMSE = {best_val ** 0.5:.4f}.")
            self._append_log("Модель готова: вкладки «3. Модель», «4. Рекомендации», «5. Оценка».")
            self.log_queue.put(("status",
                                f"Модель {name} обучена (val RMSE {best_val ** 0.5:.4f})."))
        except Exception as e:
            self._append_log(f"ОШИБКА: {e}")
        finally:
            self.log_queue.put(("done", None))

    # ------------------------------------------------------------- модель
    def _show_model_info(self):
        if self.model is None:
            return
        txt = (f"Текущая модель:   {self.model_name}\n"
               f"Параметров:       {self.model.count_parameters():,}\n"
               f"Пользователей:    {self.n_users}\n"
               f"Объектов:         {self.n_items}\n"
               f"Устройство:       CPU\n").replace(",", " ")
        self._set_text(self.model_info, txt)

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала обучите или загрузите модель.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt", initialfile="model.pt",
            filetypes=[("Модель PyTorch", "*.pt")])
        if not path:
            return
        torch.save({"name": self.model_name,
                    "n_users": self.n_users, "n_items": self.n_items,
                    "state_dict": self.model.state_dict()}, path)
        self.status.configure(text=f"Модель сохранена: {path}")
        messagebox.showinfo("Сохранено", f"Модель сохранена в файл:\n{path}")

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Файл модели", filetypes=[("Модель PyTorch", "*.pt")])
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                name = ckpt["name"]
                nu, ni = ckpt["n_users"], ckpt["n_items"]
                model = MODEL_CLASSES[name](nu, ni)
                model.load_state_dict(ckpt["state_dict"])
                self.n_users, self.n_items = nu, ni
            else:  # чистый state_dict (файл model.pt из main.py)
                name = self.model_var.get()
                model = MODEL_CLASSES[name](self.n_users or 943, self.n_items or 1682)
                model.load_state_dict(ckpt)
            model.eval()
            self.model, self.model_name = model, name
            self._show_model_info()
            self.status.configure(text=f"Модель {name} загружена из {os.path.basename(path)}.")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки модели", str(e))

    # ------------------------------------------------------------- применение
    def recommend(self):
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала обучите или загрузите модель.")
            return
        if self.processor is None:
            messagebox.showwarning("Нет данных", "Загрузите данные для сопоставления объектов.")
            return
        user_id = self.user_var.get()
        if user_id not in self.processor.user_mapping:
            messagebox.showwarning("Пользователь не найден",
                                   f"Пользователь {user_id} отсутствует в данных.")
            return
        uidx = self.processor.user_mapping[user_id]
        seen = set(self.train_data.loc[self.train_data["user_idx"] == uidx, "item_idx"])
        cand = np.array([i for i in range(self.n_items) if i not in seen])
        self.model.eval()
        with torch.no_grad():
            scores = self.model(torch.full((len(cand),), uidx, dtype=torch.long),
                                torch.tensor(cand, dtype=torch.long)).numpy()
        order = np.argsort(-scores)[: self.topn_var.get()]
        inv = {v: k for k, v in self.processor.item_mapping.items()}
        for row in self.rec_tree.get_children():
            self.rec_tree.delete(row)
        for rank, j in enumerate(order, 1):
            item_id = inv[int(cand[j])]
            title = self.item_names.get(item_id, f"Объект {item_id}")
            year = title[-5:-1] if title.endswith(")") else "—"
            name = title[:-7] if title.endswith(")") else title
            self.rec_tree.insert("", "end", values=(
                rank, name, year, f"{min(float(RATING_MAX), float(scores[j])):.2f}"))
        self.status.configure(
            text=f"Сформировано {len(order)} рекомендаций для пользователя {user_id}.")

    def export_recommendations(self):
        """Выгрузка сформированного списка рекомендаций в CSV-файл."""
        rows = self.rec_tree.get_children()
        if not rows:
            messagebox.showwarning("Нет данных",
                                   "Сначала сформируйте рекомендации.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"recommendations_user{self.user_var.get()}.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(["№", "Рекомендуемый фильм", "Год", "Прогноз оценки"])
            for row_id in rows:
                writer.writerow(self.rec_tree.item(row_id)["values"])
        self.status.configure(text=f"Рекомендации сохранены: {path}")

    # ------------------------------------------------------------- оценка
    def evaluate(self):
        if self.model is None or self.test_data is None:
            messagebox.showwarning("Не готово", "Нужны загруженные данные и обученная модель.")
            return
        self.model.eval()
        u = torch.tensor(self.test_data["user_idx"].values, dtype=torch.long)
        i = torch.tensor(self.test_data["item_idx"].values, dtype=torch.long)
        with torch.no_grad():
            preds = []
            for k in range(0, len(u), 4096):
                preds.append(self.model(u[k:k + 4096], i[k:k + 4096]).numpy())
        preds = np.clip(np.concatenate(preds), RATING_MIN, RATING_MAX)
        y = self.test_data["rating"].values
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        mae = float(np.mean(np.abs(y - preds)))
        within1 = float(np.mean(np.abs(y - preds) <= 1.0)) * 100
        txt = (f"Модель: {self.model_name}\n"
               f"Тестовая выборка: {len(y):,} оценок\n\n".replace(",", " ") +
               f"RMSE (среднеквадратичная ошибка):  {rmse:.4f}\n"
               f"MAE (средняя абсолютная ошибка):   {mae:.4f}\n"
               f"Доля прогнозов с ошибкой ≤ 1 балла: {within1:.1f} %\n")
        self._set_text(self.eval_info, txt)
        self.status.configure(text=f"Оценка выполнена: RMSE {rmse:.4f}, MAE {mae:.4f}.")


def main():
    root = tk.Tk()
    RecommenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
