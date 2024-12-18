import torch
import os
from datetime import datetime


def save_state(trainer, runsfile="artifacts/runs", prefix="training"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(runsfile, f"{prefix}_{timestamp}")

    try:
        os.makedirs(run_dir, exist_ok=True)

        weights_path = os.path.join(run_dir, "model_weights.pt")
        torch.save(trainer.model.state_dict(), weights_path)

        history_path = os.path.join(run_dir, "training_history.pt")
        torch.save(trainer.tracker.get_history(), history_path)

        with open(os.path.join(run_dir, "run_metadata.txt"), "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Epochs completed: {trainer.current_epoch}\n")
            f.write(f"Final metrics: {trainer.tracker.message()}")

        print(f"Training state saved successfully to {run_dir}")
        return run_dir

    except Exception as e:
        print(f"Error saving training state: {e}")
        raise


def load_state(run_dir, model, device=torch.device('cpu')):
    try:
        weights_path = os.path.join(run_dir, "model_weights.pt")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

        history_path = os.path.join(run_dir, "training_history.pt")
        training_history = torch.load(history_path, weights_only=False)

        print(f"Successfully loaded training state from {run_dir}")
        return training_history

    except FileNotFoundError:
        print(f"No saved state found in {run_dir}")
        raise
    except Exception as e:
        print(f"Error loading training state: {e}")
        raise


def load_recent(runsfile="artifacts/runs", model=None, device=torch.device('cpu')):
    try:
        if not os.path.exists(runsfile):
            print(f"Runs directory {runsfile} does not exist.")
            raise FileNotFoundError(f"Runs directory {runsfile} not found.")

        run_dirs = sorted(
            (os.path.join(runsfile, d) for d in os.listdir(runsfile) if os.path.isdir(os.path.join(runsfile, d))),
            key=os.path.getmtime,
            reverse=True
        )

        if not run_dirs:
            print(f"No saved runs found in {runsfile}")
            raise FileNotFoundError(f"No saved runs found in {runsfile}")

        most_recent_dir = run_dirs[0]
        print(f"Most recent run directory found: {most_recent_dir}")

        if model:
            return load_state(most_recent_dir, model, device=device)
        else:
            print(f"Model not provided. Only returning run directory path.")
            return most_recent_dir

    except Exception as e:
        print(f"Error loading most recent training state: {e}")
        raise