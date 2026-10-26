# Ablation аугментации

Validation macro-F1: среднее ± выборочное стандартное отклонение по seeds 42, 43, 44 (ddof=1).

| Condition | Аугментация on | Аугментация off |
|---|---:|---:|
| C0 | 0.9700663 ± 0.0034078 | 0.9597596 ± 0.0057848 |
| C3 | 0.9766957 ± 0.0066162 | 0.9677224 ± 0.0052049 |

## Протокол

- C0: LR `1e-3`; C3: LR `1e-5`; нового LR-поиска не выполнялось.
- Training seeds: `42, 43, 44`; `split_seed=42`.
- Checkpoint выбирался по лучшему validation macro-F1; максимум 30 epochs, early stopping patience 5.
- Аугментация off использует детерминированный `ResNet50_Weights.IMAGENET1K_V2.transforms()`; train shuffle сохранён.
- Новые запуски: CPU, `num_workers=0`, один CPU thread, float32, текущий shared trainer.

## Источники

- Полная агрегация: `artifacts/ablation_noaug/summary/ablation.csv`.
- C0-on: `deliverables/c0/evaluation/lr_1e-3/metrics.csv`.
- C3-on: `deliverables/c3/evaluation/lr_1e-5/metrics.csv`.
- C0/C3-off histories, configs, metadata и checkpoints: `artifacts/ablation_noaug/runs/`.

## Ограничения сопоставимости

C0-on был обучен на CPU с Python 3.14.6 / torch 2.14.1 при историческом dirty commit `155bf1b`; C3-on — на Tesla T4 с Python 3.13.15 / torch 2.11.0+cu128 при commit `37f1e66`; оба off-набора выполнены локально на CPU текущим trainer из commit `367b6c2`. Исторические same-seed head hashes C0/C3 различаются, поэтому сравнение не считается идеально контролируемой изоляцией аугментации. Split membership, labels и row order совпадают; различие byte hashes C3 объясняется CRLF/LF.

## Интерпретация

В этих шести запусках отключение аугментации изменило средний validation macro-F1 на -0.010307 для C0 и на -0.008973 для C3; это описательное сравнение не доказывает статистическую значимость или причинность.
