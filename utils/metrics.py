from utils.per import PhonemeErrorRate


class MetricsModule:
    def __init__(self, set_name, device, has_articulatory_heads=False) -> None:
        self.device = device
        self.set_name = set_name
        self.has_articulatory_heads = has_articulatory_heads

        dict_metrics = {}

        # Phoneme error rate
        dict_metrics["per"] = PhonemeErrorRate().to(device)

        # Average articulatory error rate (if applicable)
        if has_articulatory_heads:
            dict_metrics["avg_aer"] = PhonemeErrorRate().to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y, articulatory_predictions=None, articulatory_targets=None):
        self.dict_metrics["per"](x, y)

        if (self.has_articulatory_heads and
                articulatory_predictions is not None and
                articulatory_targets is not None):

            all_art_preds = []
            all_art_targets = []

            feature_names = sorted(articulatory_predictions.keys())

            for feature_name in feature_names:
                if feature_name in articulatory_predictions and feature_name in articulatory_targets:
                    all_art_preds.extend(articulatory_predictions[feature_name])
                    all_art_targets.extend(articulatory_targets[feature_name])

            if all_art_preds and all_art_targets:
                self.dict_metrics["avg_aer"](all_art_preds, all_art_targets)

    def log_metrics(self, name, pl_module):
        """
        Compute and log all metrics

        Args:
            name: Prefix for logging (e.g., "train/", "val/")
            pl_module: PyTorch Lightning module for logging
        """
        metrics_to_print = {}

        for k, m in self.dict_metrics.items():
            # Compute metric
            metric = m.compute()
            pl_module.log(name + k, metric)

            # Store for printing
            metrics_to_print[k] = metric.item() if hasattr(metric, 'item') else metric

            # Reset metric for next epoch
            m.reset()
            m.to(self.device)

        # Print to terminal
        if self.has_articulatory_heads:
            epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else '?'
            print(
                f"\n[{self.set_name.upper()} Epoch {epoch}] PER: {metrics_to_print['per']:.4f} | Avg AER: {metrics_to_print['avg_aer']:.4f}")
        else:
            epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else '?'
            print(f"\n[{self.set_name.upper()} Epoch {epoch}] PER: {metrics_to_print['per']:.4f}")