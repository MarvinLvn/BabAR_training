from utils.per import PhonemeErrorRate


class MetricsModule:
    def __init__(self, set_name, device, articulatory_feature_names=None) -> None:
        self.device = device
        self.set_name = set_name
        if articulatory_feature_names is None:
            self.articulatory_feature_names = []
        else:
            self.articulatory_feature_names = articulatory_feature_names

        dict_metrics = {}

        # Phoneme error rate
        dict_metrics["per"] = PhonemeErrorRate().to(device)

        # Average articulatory error rate (if applicable)
        for feature_name in self.articulatory_feature_names:
            dict_metrics[f"aer_{feature_name}"] = PhonemeErrorRate().to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y, articulatory_predictions=None, articulatory_targets=None):
        self.dict_metrics["per"](x, y)

        if articulatory_predictions is not None and articulatory_targets is not None:
            for feature_name in self.articulatory_feature_names:
                preds = articulatory_predictions[feature_name]
                targets = articulatory_targets[feature_name]
                self.dict_metrics[f"aer_{feature_name}"](preds, targets)

    def log_metrics(self, name, pl_module):
        """
        Compute and log all metrics

        Args:
            name: Prefix for logging (e.g., "train/", "val/")
            pl_module: PyTorch Lightning module for logging
        """
        metrics_to_print = {}
        per_metric = self.dict_metrics["per"].compute()
        pl_module.log(name + "per", per_metric)
        metrics_to_print['per'] = per_metric.item() if hasattr(per_metric, 'item') else per_metric
        self.dict_metrics["per"].reset()

        # Print to terminal
        if len(self.articulatory_feature_names) != 0:
            feature_errors = []
            for feature_name in self.articulatory_feature_names:
                metric_key = f"aer_{feature_name}"
                aer_metric = self.dict_metrics[metric_key].compute()
                feature_errors.append(aer_metric)
                self.dict_metrics[metric_key].reset()

            avg_aer = sum(feature_errors) / len(feature_errors)
            pl_module.log(name + "avg_aer", avg_aer)
            metrics_to_print['avg_aer'] = avg_aer.item() if hasattr(avg_aer, 'item') else avg_aer

        if 'avg_aer' in metrics_to_print:
            epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else '?'
            print(
                f"\n[{self.set_name.upper()} Epoch {epoch}] PER: {metrics_to_print['per']:.4f} | Avg AER: {metrics_to_print['avg_aer']:.4f}")
        else:
            epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else '?'
            print(f"\n[{self.set_name.upper()} Epoch {epoch}] PER: {metrics_to_print['per']:.4f}")