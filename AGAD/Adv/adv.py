import torch as th
class ADV:
    def __init__(self, model, eps=0.1):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}
        self.last_r_at = 0

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                param.data.add_(self.last_r_at)
                param.data = self.project(name, param.data)
                self.last_r_at = self.last_r_at + self.eps * param.grad.sign()

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if th.norm(r) > self.eps:
            r = self.eps * r / th.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
