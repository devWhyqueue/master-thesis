import argparse
import json
import logging
import subprocess

def load_config(config_path: str) -> dict[str, str]:
    """Lädt die Konfiguration aus einer JSON-Datei."""
    try:
        with open(config_path, "r", encoding="utf-8") as f: return json.load(f)
    except FileNotFoundError:
        logging.warning("Konfigurationsdatei %s nicht gefunden.", config_path)
        return {}

def build_prefix(config: dict[str, str], local: bool, no_container: bool) -> list[str]:
    """Erstellt den Präfix-Befehl (entweder Apptainer oder direkt python3)."""
    if local and no_container: return ["python3"]
    return ["apptainer", "run", "-B", "/home/space:/home/space:rw", config.get("environment_sif", ""), "python3"]

def submit_slurm(cmd: list[str], job_name: str, log_path: str, config: dict[str, str], dry_run: bool) -> None:
    """Erstellt ein SLURM-Skript und reicht es bei sbatch ein."""
    working_dir = config.get("working_dir", "")
    sh_script = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}\n#SBATCH --partition=cpu-2h\n"
        f"#SBATCH --ntasks-per-node=8\n#SBATCH --output={log_path}\n#SBATCH -D {working_dir}\n\n"
        f"{' '.join(cmd)}\n"
    )
    if dry_run: logging.info("Dry-run (SLURM %s):\n%s", job_name, sh_script)
    else:
        logging.info("Reiche Job %s bei SLURM ein...", job_name)
        subprocess.run(["sbatch"], input=sh_script, text=True, check=True)

def execute(cmd: list[str], job_name: str, log_path: str, config: dict[str, str], local: bool, dry_run: bool) -> None:
    """Führt einen Befehl aus (lokal oder reicht ihn an SLURM weiter)."""
    if local:
        if dry_run: logging.info("Dry-run (lokal): %s", " ".join(cmd))
        else:
            logging.info("Führe lokal aus: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
    else: submit_slurm(cmd, job_name, log_path, config, dry_run)

def handle_balanced(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Verarbeitet den Befehl zur ausbalancierten Stichprobenziehung."""
    cmd = build_prefix(config, args.local, args.no_container) + [
        "sample_TCGA_UT_balanced.py", f"--dataset-path={config.get('dataset_path', '')}",
        f"--file-save-path={config.get('balanced_dataset_csv', '')}",
        f"--n-slides-per-class={args.n_slides_per_class}", f"--n-patches-per-slide={args.n_patches_per_slide}"
    ]
    execute(cmd, "sample_balanced", "logs/sampling/sample_balanced%j.out", config, args.local, args.dry_run)

def handle_imbalanced(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Verarbeitet den Befehl zur unbalancierten Stichprobenziehung."""
    bal_csv, imb_dir = config.get("balanced_dataset_csv", ""), config.get("imbalanced_dataset_dir", "")
    prefix = build_prefix(config, args.local, args.no_container)
    params = [round(i / 10.0, 1) for i in range(14)] if args.sweep else [args.parameter]
    for p in params:
        out = f"{imb_dir}/TCGA-UT_imbalanced_parameter={p}_dataset_size=500_seed=0/imbalanced_dataset.csv"
        cmd = prefix + [
            "sample_TCGA_UT_imbalanced.py", f"--balanced-dataset-path={bal_csv}", f"--file-save-path={out}",
            f"--parameter={p}", "--dataset-size=500", "--sample-balanced-validation", "--n-slides-per-class=10",
            "--visualize", '--overflow-strategy="redistribute"', "--n-regions-per-slide=3",
            "--n-patches-per-region=10", "--store-class-names"
        ]
        execute(cmd, "sample", "logs/sampling/sample_imbalanced%j.out", config, args.local, args.dry_run)

def train_mlp(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Führt das MLP-Training aus."""
    imb_dir, res_dir = config.get("imbalanced_dataset_dir", ""), config.get("results_dir", "")
    prefix = build_prefix(config, args.local, args.no_container)
    params = [round(i / 10.0, 1) for i in range(14)] if args.sweep else [args.parameter]
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    for p in params:
        for s in seeds:
            ds = f"{imb_dir}/TCGA-UT_imbalanced_parameter={p}_dataset_size=500_seed=0/imbalanced_dataset.csv"
            val = f"{imb_dir}/TCGA-UT_imbalanced_parameter={p}_dataset_size=500_seed=0/validation_dataset.csv"
            out = f"{res_dir}/param={p}/seed={s}"
            cmd = prefix + [
                "train.py", f"--dataset-structure-path={ds}", f"--validation-dataset-structure-path={val}",
                f"--feature-path={config.get('feature_path', '')}", "--preload-features", f"--results-save-path={out}",
                '--device="cpu"', "--learning-rate=0.001", "--n-epochs=50", '--loss="cross_entropy"',
                '--alpha="uniform"', "--batch-balancing", f"--seed={s}", "--visualize",
                f"--class-names-path={config.get('class_names_path', '')}"
            ]
            execute(cmd, "train", "logs/training/train%j.out", config, args.local, args.dry_run)

def train_knn(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Führt das KNN-Training aus."""
    bal_csv, res_dir = config.get("balanced_dataset_csv", ""), config.get("results_dir", "")
    prefix = build_prefix(config, args.local, args.no_container)
    ks = [3, 9, 27] if args.sweep else [args.k]
    val_csv = config.get("validation_dataset_csv", bal_csv)
    for k in ks:
        cmd = prefix + [
            "train.py", f"--dataset-structure-path={bal_csv}", f"--validation-dataset-structure-path={val_csv}",
            f"--feature-path={config.get('feature_path', '')}", "--preload-features",
            f"--results-save-path={res_dir}/results_knn/k={k}/", '--device="cpu"', '--model="knn"', f"--k={k}",
            "--visualize", f"--class-names-path={config.get('class_names_path', '')}"
        ]
        execute(cmd, "train_knn", "logs/training/train_knn%j.out", config, args.local, args.dry_run)

def train_ncc(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Führt das NCC-Training aus."""
    imb_dir, res_dir = config.get("imbalanced_dataset_dir", ""), config.get("results_dir", "")
    prefix = build_prefix(config, args.local, args.no_container)
    params = [round(i / 10.0, 1) for i in range(14)] if args.sweep else [args.parameter]
    for p in params:
        ds = f"{imb_dir}/TCGA-UT_imbalanced_parameter={p}_dataset_size=500_seed=0/imbalanced_dataset.csv"
        val = f"{imb_dir}/TCGA-UT_imbalanced_parameter={p}_dataset_size=500_seed=0/validation_dataset.csv"
        out = f"{res_dir}/results_ncc/param={p}/"
        cmd = prefix + [
            "train.py", f"--dataset-structure-path={ds}", f"--validation-dataset-structure-path={val}",
            f"--feature-path={config.get('feature_path', '')}", "--preload-features", f"--results-save-path={out}",
            '--device="cpu"', '--model="ncc"', "--visualize", f"--class-names-path={config.get('class_names_path', '')}"
        ]
        execute(cmd, "train_ncc", "logs/training/train_ncc%j.out", config, args.local, args.dry_run)

def handle_train(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Delegiert das Training an die modellspezifischen Funktionen."""
    if args.model == "mlp": train_mlp(args, config)
    elif args.model == "knn": train_knn(args, config)
    else: train_ncc(args, config)

def visualize_standard(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Führt Standardvisualisierungen aus."""
    res_dir, viz_dir = config.get("results_dir", ""), config.get("visualization_dir", "")
    cmd = build_prefix(config, args.local, args.no_container) + [
        "visualizations.py", "--plot-types", "scatter_accuracies_of_two_parameters",
        "difference_confusion_matrix", "confusion_matrix", "--results-paths",
        f"{res_dir}/results_batch_balancing/", f"{res_dir}/results_batch_balancing/",
        "--parameters", "1.0", "0.0", "--parameter-name", "param", "--visualization-save-path", f"{viz_dir}/"
    ]
    execute(cmd, "viz", "logs/viz/viz%j.out", config, args.local, args.dry_run)

def visualize_point_plot(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Führt die Point-Plot-Visualisierung zum Methodenvergleich aus."""
    res_dir, viz_dir = config.get("results_dir", ""), config.get("visualization_dir", "")
    cmd = build_prefix(config, args.local, args.no_container) + [
        "visualizations.py", "--plot-types", "point_plot_compare_methods", "--results-paths",
        f"{res_dir}/results_cross_entropy_inverse_class_frequency", f"{res_dir}/results_batch_balancing/",
        f"{res_dir}/results_focal_loss_inverse_class_frequency/", f"{res_dir}/results_focal_loss_uniform/",
        f"{res_dir}/results_original_class_size_order/", f"{res_dir}/results_original_class_size_order/",
        "--parameters", "1.0", "1.0", "1.0", "1.0", "1.0", "0.0", "--parameter-name", "param",
        "--methods", "Weighted Cross Entropy", "Batch Balancing", "Weighted Focal Loss", "Unweighted Focal Loss",
        "Vanilla", "Balanced", "--visualization-save-path", f"{viz_dir}/"
    ]
    execute(cmd, "viz", "logs/viz/viz%j.out", config, args.local, args.dry_run)

def handle_visualize(args: argparse.Namespace, config: dict[str, str]) -> None:
    """Delegiert die Visualisierung basierend auf dem Typ."""
    if args.type == "standard": visualize_standard(args, config)
    else: visualize_point_plot(args, config)

def add_balanced_args(subparsers: argparse._SubParsersAction) -> None:
    """Fügt Argumente für balanced-sampling hinzu."""
    p = subparsers.add_parser("sample-balanced", help="Ausbalancierte Version von TCGA-UT samplen")
    p.add_argument("--n-slides-per-class", type=int, default=100, help="Anzahl der Slides pro Klasse")
    p.add_argument("--n-patches-per-slide", type=int, default=30, help="Anzahl der Patches pro Slide")

def add_imbalanced_args(subparsers: argparse._SubParsersAction) -> None:
    """Fügt Argumente für imbalanced-sampling hinzu."""
    p = subparsers.add_parser("sample-imbalanced", help="Unbalancierte Version von TCGA-UT samplen")
    p.add_argument("--sweep", action="store_true", help="Parameter-Sweep von 0.0 bis 1.3 durchführen")
    p.add_argument("--parameter", type=float, default=1.0, help="Imbalance-Parameter (ignoriert bei Sweep)")

def add_train_args(subparsers: argparse._SubParsersAction) -> None:
    """Fügt Argumente für das Training hinzu."""
    p = subparsers.add_parser("train", help="Modell trainieren")
    p.add_argument("model", choices=["mlp", "knn", "ncc"], help="Modelltyp")
    p.add_argument("--sweep", action="store_true", help="Parameter-Sweep durchführen")
    p.add_argument("--parameter", type=float, default=1.0, help="Imbalance-Parameter (ignoriert bei Sweep)")
    p.add_argument("--seed", type=int, default=0, help="Zufallssaat für MLP (ignoriert bei Sweep)")
    p.add_argument("--k", type=int, default=9, help="Anzahl k für KNN (ignoriert bei Sweep)")

def add_visualize_args(subparsers: argparse._SubParsersAction) -> None:
    """Fügt Argumente für Visualisierungen hinzu."""
    p = subparsers.add_parser("visualize", help="Visualisierungen erstellen")
    p.add_argument("type", choices=["standard", "point-plot"], help="Visualisierungstyp")

def create_parser() -> argparse.ArgumentParser:
    """Erstellt den Haupt-Argument-Parser."""
    parser = argparse.ArgumentParser(description="Zentralisiertes Skript zum Ausführen von TCGA-UT-Jobs.")
    parser.add_argument("--config", default="config.json", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--local", action="store_true", help="Führe Jobs lokal anstatt über SLURM aus")
    parser.add_argument("--no-container", action="store_true", help="Führe python direkt ohne Apptainer aus")
    parser.add_argument("--dry-run", action="store_true", help="Zeige Befehle an, ohne sie auszuführen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_balanced_args(subparsers)
    add_imbalanced_args(subparsers)
    add_train_args(subparsers)
    add_visualize_args(subparsers)
    return parser

def main() -> None:
    """Hauptfunktion des Skripts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "sample-balanced": handle_balanced(args, config)
    elif args.command == "sample-imbalanced": handle_imbalanced(args, config)
    elif args.command == "train": handle_train(args, config)
    elif args.command == "visualize": handle_visualize(args, config)

if __name__ == "__main__":
    main()
