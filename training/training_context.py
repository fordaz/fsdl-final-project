from collections import defaultdict
import mlflow


class TrainingContext:
    def __init__(self, args):
        self.all_metrics = defaultdict(list)
        self.stop_early = False
        self.early_stopping_step = 0
        self.early_stopping_best_val = float("inf")
        self.early_stopping_criteria = args.early_stopping_criteria

    def append_metrics(self, metrics):
        for m_name, m_val in metrics.items():
            self.all_metrics[m_name].append(m_val)

    def update(self, model, epoch, saved_model_fname):
        saved_model_fname = f"{saved_model_fname}_{epoch}"
        # Save one model at least
        if epoch == 0:
            mlflow.pytorch.save_model(pytorch_model=model, path=saved_model_fname)
            self.stop_early = False

        # Save model if performance improved
        elif epoch >= 1:
            loss_tm1, loss_t = self.all_metrics["val_loss"][-2:]

            # If loss worsened
            if loss_t >= loss_tm1:
                # Update step
                self.early_stopping_step += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.early_stopping_best_val:
                    mlflow.pytorch.save_model(
                        pytorch_model=model, path=saved_model_fname
                    )
                    self.early_stopping_best_val = loss_t

                # Reset early stopping step
                self.early_stopping_step = 0

            # Stop early ?
            self.stop_early = self.early_stopping_step >= self.early_stopping_criteria
