from torch.optim.lr_scheduler import LambdaLR


class TriStageLR(LambdaLR):
    """
    Tri-stage schedule: warmup until warmup_ratio → constant until constant_ratio → linear decay for the remaining steps
    """

    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, constant_ratio=0.4, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.constant_ratio = constant_ratio
        self.decay_ratio = 1.0 - warmup_ratio - constant_ratio  # Remaining percentage

        self.warmup_steps = int(warmup_ratio * total_steps)
        self.constant_steps = int(constant_ratio * total_steps)
        self.decay_start = self.warmup_steps + self.constant_steps

        # Validation
        if warmup_ratio + constant_ratio >= 1.0:
            raise ValueError(f"warmup_ratio ({warmup_ratio}) + constant_ratio ({constant_ratio}) must be < 1.0")

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, self.warmup_steps))
            elif current_step < self.decay_start:
                # Constant phase
                return 1.0
            else:
                # Linear decay phase
                decay_steps = self.total_steps - self.decay_start
                return max(0.0, float(self.total_steps - current_step) / float(max(1, decay_steps)))

        super().__init__(optimizer, lr_lambda, last_epoch)