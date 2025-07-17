# Notes from what the exporter is printing

tool to compare models

```
python scripts/model_validation.py 
```

The exporter in the SKRL is merging the model so ONNX can use. Might not be right since the model outputs are widely different.

It is making a deepcopy of the NNs module as created by SKRL Runner

```python
# copy policy parameters
self._nn = copy.deepcopy(policy)

# Need to import the template model torch hook
self.model = MyModel(self._nn)
```

Then it is running the MyModel(self._nn) so it merges all the NN modules and then create a Hook forward() that the ONNX model can use.

```python
print(f"printing the self._nn: {self._nn}")
print(f"printing the self.model: {self.model}")
# print(f"Zeros from the net container:{self._nn.net_container[0].in_features}")
# obs = torch.zeros(1, 4) # self._nn.net_container[0].in_features
obs = torch.zeros(1, self._nn.net_container[0].in_features)
# print(obs)
torch.onnx.export(
    self.model, # self -- this should be wrong -- self.model -- fail self._nn.forward(self._nn)
    obs, # model input (or a tuple for multiple inputs)
    os.path.join(path, filename),
    export_params=True,
    opset_version=11,
    verbose=self.verbose,
    input_names=["obs"],
    output_names=["actions"], # "taken_actions"
    dynamic_axes={},
)
```

```python
class MyModel(torch.nn.Module):
    def __init__(self, policy):
        super(MyModel, self).__init__()
        self.policy = policy
        # # Define the sequential part
        # self.sequential = policy.net_container 
        # # Define the linear part
        # self.linear = policy.policy_layer

    def forward(self, x):
        for name, module in self.policy._modules.items():
            print(f"Submodule name: {name}, Submodule: {module}") 
            if name == "value_layer":
                continue
            else:
                x = module(x)
        return x
```

```bash
logger.warn(
[skrl:INFO] Environment wrapper: Isaac Lab (single-agent)
[skrl:INFO] Seed: 42
==================================================
Shared model (roles): ['policy', 'value']
==================================================

class SharedModel(GaussianMixin,DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
        )
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=""):
        if role == "policy":
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            net = self.net_container(states)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                net = self.net_container(states)
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
    
--------------------------------------------------
==================================================
Observations for Agent_id agent Shared model (roles): ['policy', 'value']
==================================================

Box(-inf, inf, (8,), float32)
--------------------------------------------------
[INFO] Loading model checkpoint from: /home/goat/Documents/GitHub/renanmb/leatherback/logs/skrl/leatherback_direct/2025-07-15_16-03-28_ppo_torch/checkpoints/best_agent.pt
printing the self._nn: SharedModel(
  (net_container): Sequential(
    (0): Linear(in_features=8, out_features=32, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=32, out_features=32, bias=True)
    (3): ELU(alpha=1.0)
  )
  (policy_layer): Linear(in_features=32, out_features=2, bias=True)
  (value_layer): Linear(in_features=32, out_features=1, bias=True)
)
printing the self.model: MyModel(
  (policy): SharedModel(
    (net_container): Sequential(
      (0): Linear(in_features=8, out_features=32, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ELU(alpha=1.0)
    )
    (policy_layer): Linear(in_features=32, out_features=2, bias=True)
    (value_layer): Linear(in_features=32, out_features=1, bias=True)
  )
)
Submodule name: net_container, Submodule: Sequential(
  (0): Linear(in_features=8, out_features=32, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=32, out_features=32, bias=True)
  (3): ELU(alpha=1.0)
)
Submodule name: policy_layer, Submodule: Linear(in_features=32, out_features=2, bias=True)
Submodule name: value_layer, Submodule: Linear(in_features=32, out_features=1, bias=True)
env_ids before function call: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
       device='cuda:0'), type: <class 'torch.Tensor'>, device: cuda:0
```

```bash
printing self _OnnxPolicyExporter(
  (_nn): SharedModel(
    (net_container): Sequential(
      (0): Linear(in_features=8, out_features=32, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ELU(alpha=1.0)
    )
    (policy_layer): Linear(in_features=32, out_features=2, bias=True)
    (value_layer): Linear(in_features=32, out_features=1, bias=True)
  )
  (model): MyModel(
    (policy): SharedModel(
      (net_container): Sequential(
        (0): Linear(in_features=8, out_features=32, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(in_features=32, out_features=32, bias=True)
        (3): ELU(alpha=1.0)
      )
      (policy_layer): Linear(in_features=32, out_features=2, bias=True)
      (value_layer): Linear(in_features=32, out_features=1, bias=True)
    )
  )
)
```

```bash
printing self _OnnxPolicyExporter(
  (actor): SharedModel(
    (net_container): Sequential(
      (0): Linear(in_features=8, out_features=32, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ELU(alpha=1.0)
    )
    (policy_layer): Linear(in_features=32, out_features=2, bias=True)
    (value_layer): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

## RSL_RL stuff

RSL_RL ONNX exporter, printing the self

```bash
printing self _OnnxPolicyExporter(
  (actor): Sequential(
    (0): Linear(in_features=48, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
  (normalizer): Identity()
)
```

```bash
printing the policy_nn: ActorCritic(
  (actor): Sequential(
    (0): Linear(in_features=48, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
  (critic): Sequential(
    (0): Linear(in_features=48, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

Spot flat trained with SKRL

```bash
printing self _OnnxPolicyExporter(
  (actor): Sequential(
    (0): Linear(in_features=48, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
  (normalizer): Identity()
)
```