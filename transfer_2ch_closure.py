import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_mnist import MNIST_Net, test                   # Your trained MLP from previous script
import math
import torch.nn.functional as F
from torch.optim import AdamW, LBFGS, lr_scheduler
import numpy as np
from torch import Tensor

class Clos(nn.Module):
    def __init__(self, in_features=768, out_features=None, channel=3, switches=None, bias=True, middle_switch_multiplier=4):
        """switches={int: bin, b1, b2, b3, bout}
        in_features=768, out_features=768,
        channel=3,  bias=True"""
        super(Clos, self).__init__()

        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.channel = channel
        self.bias = bias
        self.middle_switch_multiplier = middle_switch_multiplier
        self.switches = {}

        # orolt garaltiin tootsoo hiine, custom input ugvul update hiine
        self.find_factors()
        
        if switches is not None:
            self.switches.update(switches)

        # weightuud
        k = 1.0 / math.sqrt(in_features)       
        self.weight1 = nn.Parameter(torch.Tensor( 
                        self.switches['bin'], 
                        self.switches['b1'], 
                        self.switches['b2']
                        ))

        self.weight2 = nn.Parameter(torch.Tensor(   
                        self.switches['b1'],
                        self.switches['b2'],
                        self.switches['b3'],
                        ))

        self.weight3 = nn.Parameter(torch.Tensor(
                        self.switches['b2'],
                        self.switches['b3'], 
                        self.switches['bout']
                        ))

        # bias
        if self.bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.switches['b1']))
            self.bias2 = nn.Parameter(torch.Tensor(self.switches['b2']))
            self.bias3 = nn.Parameter(torch.Tensor(self.switches['b3']))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
            self.register_parameter('bias3', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias1, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias2, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight3)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias3, -bound, bound)

    def find_factors(self):
        for i in range(int(math.sqrt(self.in_features)), 0, -1):
            if self.in_features % i == 0:
                self.switches['bin'] = i
                self.switches['b1'] = self.in_features // i
                break
        
        for i in range(int(math.sqrt(self.out_features)), 0, -1):
            if self.out_features % i == 0:
                self.switches['bout'] = i
                self.switches['b3'] = self.out_features // i
                break
        self.switches['b2'] = self.middle_switch_multiplier * self.switches['bin'] #Middle switch multiplier 4

    def __repr__(self):
        return (
            f"Clos(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias}, "\
            f"bin={self.switches['bin']}, "\
            f"b1={self.switches['b1']}, "\
            f"b2={self.switches['b2']}, "\
            f"b3={self.switches['b3']}, "\
            f"bout={self.switches['bout']}, "\
            f"channel={self.channel})"
        )

    def channel2(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, self.switches['bin'], self.switches['b1'])

        if self.bias:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1) + self.bias1
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2) + self.bias2
            x = torch.einsum('bnm,mro->bor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1)
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2)
            x = torch.einsum('bnm,mro->bor', x, self.weight3)

        return x.reshape(b, -1)

    # Channel=3: for attention-style [B, C, H] (e.g., batch, heads, dim)
    def channel3(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        x = x.view(b, c, self.switches['bin'], self.switches['b1'])

        if self.bias:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1) + self.bias1
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2) + self.bias2
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1)
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2)
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3)

        return x.reshape(b, c, -1)
    def forward(self, input: Tensor) -> Tensor:
        if self.channel == 2:
            return self.channel2(input)
        elif self.channel == 3:
            return self.channel3(input)

def transfer_fc_to_clos_fc1aware(
    fc: nn.Linear,
    channel: int = 2,
    max_steps: int = 20000,
    lr: float = 2e-3,
    middle_switch_multiplier: int = 4,
    probe_rand: int = 32768,
    probe_batch: int = 1024,
    gate_margin: float = 0.25,
    lam_eye: float = 5.0,
    lam_gate: float = 1.0,
    seed: int | None = None,
    verbose: bool = True,
):
    device = next(fc.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    in_f, out_f = fc.in_features, fc.out_features
    clos = Clos(in_features=in_f, out_features=out_f, channel=channel, bias=True,
                middle_switch_multiplier=middle_switch_multiplier).to(device)

    fc.eval()
    clos.train()

    # Probes: include zeros + eye + random
    with torch.no_grad():
        eye = torch.eye(in_f, device=device)
        z = torch.zeros(256, in_f, device=device)
        Xr = torch.randn(probe_rand, in_f, device=device)
        X = torch.cat([z, eye, -eye, Xr], dim=0)
        T = fc(X)  # teacher preactivations

    def clos_forward(x):
        if channel == 2:
            return clos(x)
        else:
            return clos(x.unsqueeze(1)).squeeze(1)

    opt = AdamW(clos.parameters(), lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    # for reporting
    with torch.no_grad():
        T_eye = fc(eye)

    for step in range(max_steps):
        # batch sample
        k_struct = min(probe_batch // 2, (256 + 2*in_f))  # zeros + (eye/-eye)
        k_rand = probe_batch - k_struct
        struct_idx = torch.randint(0, 256 + 2*in_f, (k_struct,), device=device)
        rand_idx = torch.randint(256 + 2*in_f, X.size(0), (k_rand,), device=device)
        idx = torch.cat([struct_idx, rand_idx], dim=0)

        xb = X[idx]
        tb = T[idx]

        opt.zero_grad(set_to_none=True)
        sb = clos_forward(xb)

        # 1) match preactivations
        loss_mse = F.mse_loss(sb, tb)

        # 2) preserve ReLU gates (sign pattern) with a margin
        # teacher sign: +1 if tb>0 else -1
        sign = torch.where(tb > 0, torch.ones_like(tb), -torch.ones_like(tb))
        # want sign * sb >= margin  => hinge = relu(margin - sign*sb)
        loss_gate = F.relu(gate_margin - sign * sb).mean()

        # 3) extra anchor on eye mapping
        s_eye = clos_forward(eye)
        loss_eye = F.mse_loss(s_eye, T_eye)

        loss = loss_mse + lam_gate * loss_gate + lam_eye * loss_eye

        loss.backward()
        torch.nn.utils.clip_grad_norm_(clos.parameters(), 1.0)
        opt.step()
        sched.step()

        if verbose and (step % 1000 == 0 or step == max_steps - 1):
            with torch.no_grad():
                eye_mse = F.mse_loss(clos_forward(eye), T_eye).item()
                gate_violation = (F.relu(gate_margin - sign * sb) > 0).float().mean().item()
            print(f"step {step:5d} | loss {loss.item():.4e} | eye_mse {eye_mse:.4e} | gate_viol {gate_violation:.3f}")

    return clos



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Load pre-trained MLP
    model = MNIST_Net().to(device)
    model.load_state_dict(torch.load("/workspace/Data2/embedder_own/clos/mnist_best.pth", map_location=device))
    print("Original model loaded (mnist_best.pth)")
    # Original fc1 to use as teacher for CLOS distillation
    fc = model.fc2
    # Test loader (same as training script)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,
                             num_workers=4, pin_memory=True)
    accuracy_original = test(model, test_loader, use_amp=True, device=device)
    print("Accuracy of original Linear layer:", accuracy_original)
    # Start searching for a good CLOS replacement
    best_acc = 0.0
    rnd_seed = np.random.randint(0, 1000000)
    for i in range(5):
        clos = transfer_fc_to_clos_fc1aware(
            fc=fc,
            channel=2,
            max_steps=3000,
            lr=2e-3,
            middle_switch_multiplier=4,
            verbose=True,
            seed=rnd_seed,  # IMPORTANT: allow different random init if you rerun
        )
        model.fc1 = clos
        model.eval()
        accuracy_clos = test(model, test_loader, use_amp=True, device=device)
        print(f"Accuracy after replacing with CLOS {i}th:", accuracy_clos)
        if accuracy_clos > best_acc:
            best_acc = accuracy_clos
            torch.save(clos.state_dict(), "clos_784_best_test.pth")
            print(f"  → NEW BEST! {best_acc:.3f}% → saved to clos_784_best_test.pth\n")

if __name__ == "__main__":
    main()