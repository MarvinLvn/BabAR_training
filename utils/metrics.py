from utils.per import PhonemeErrorRate


class MetricsModule:
    def __init__(self, set_name, device) -> None:
        self.device = device
        self.set_name = set_name

        dict_metrics = {}
        dict_metrics["per"] = PhonemeErrorRate().to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):
        self.dict_metrics["per"](x, y)

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

        epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else '?'
        print(f"\n[{self.set_name.upper()} Epoch {epoch}] PER: {metrics_to_print['per']:.4f}")