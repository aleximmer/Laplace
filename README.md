<div align="center">
 <img src="https://raw.githubusercontent.com/AlexImmer/Laplace/main/logo/laplace_logo.png" alt="Laplace" width="300"/>

![pytest](https://github.com/aleximmer/laplace/actions/workflows/pytest-default.yml/badge.svg)
![lint](https://github.com/aleximmer/laplace/actions/workflows/lint-ruff.yml/badge.svg)
![format](https://github.com/aleximmer/laplace/actions/workflows/format-ruff.yml/badge.svg)

</div>

The laplace package facilitates the application of Laplace approximations for entire neural networks, subnetworks of neural networks, or just their last layer.
The package enables posterior approximations, marginal-likelihood estimation, and various posterior predictive computations.
The library documentation is available at [https://aleximmer.github.io/Laplace](https://aleximmer.github.io/Laplace).

There is also a corresponding paper, [_Laplace Redux — Effortless Bayesian Deep Learning_](https://arxiv.org/abs/2106.14806), which introduces the library, provides an introduction to the Laplace approximation, reviews its use in deep learning, and empirically demonstrates its versatility and competitiveness. Please consider referring to the paper when using our library:

```bibtex
@inproceedings{laplace2021,
  title={Laplace Redux--Effortless {B}ayesian Deep Learning},
  author={Erik Daxberger and Agustinus Kristiadi and Alexander Immer
          and Runa Eschenhagen and Matthias Bauer and Philipp Hennig},
  booktitle={{N}eur{IPS}},
  year={2021}
}
```

The [code](https://github.com/runame/laplace-redux) to reproduce the experiments in the paper is also publicly available; it provides examples of how to use our library for predictive uncertainty quantification, model selection, and continual learning.

> [!IMPORTANT]
> As a user, one should not expect Laplace to work automatically.
> That is, one should experiment with different Laplace's options
> (hessian_factorization, prior precision tuning method, predictive method, backend,
> etc!). Try looking at various papers that use Laplace for references on how to
> set all those options depending on the applications/problems at hand.

## Table of contents

1. [Setup](#setup)
2. [Example usage](#example-usage)
   1. [Simple usage](#simple-usage)
   2. [Marginal likelihood](#marginal-likelihood)
   3. [Laplace on LLM](#laplace-on-llm)
   4. [Subnetwork Laplace](#subnetwork-laplace)
   5. [Serialization](#serialization)
3. [Structure](#structure)
4. [Extendability](#extendability)
5. [When to use which backend?](#when-to-use-which-backend)
6. [Contributing](#contributing)
7. [References](#references)

## Setup

> [!IMPORTANT]
> We assume Python >= 3.9 since lower versions are [(soon to be) deprecated](https://devguide.python.org/versions/).
> PyTorch version 2.0 and up is also required for full compatibility.

To install laplace with `pip`, run the following:

```bash
pip install laplace-torch
```

Additionally, if you want to use the `asdfghjkl` backend, please install it via:

```bash
pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl
```

### Setup dev environment

For development purposes, e.g. if you would like to make contributions, follow
the following steps:

**With `uv`**

1. Install [`uv`](https://github.com/astral-sh/uv), e.g. `pip install --upgrade uv`
2. Then clone this repository and install the development dependencies:

```bash
git clone git@github.com:aleximmer/Laplace.git
uv sync --all-extras
```

3. `laplace-torch` is now available in editable mode, e.g. you can run:

```bash
uv run python examples/regression_example.py

# Or, equivalently:
source .venv/bin/activate
python examples/regression_example.py
```

**With `pip`**

```bash
git clone git@github.com:aleximmer/Laplace.git

# Recommended to create a virtualenv before the following step
pip install -e ".[dev]"

# Run as usual, e.g.
python examples/regression_examples.py
```

> [!NOTE]
> See [contributing guideline](#contributing).
> We're looking forward to your contributions!

## Example usage

### Simple usage

In the following example, a pre-trained model is loaded,
then the Laplace approximation is fit to the training data
(using a diagonal Hessian approximation over all parameters),
and the prior precision is optimized with cross-validation `"gridsearch"`.
After that, the resulting LA is used for prediction with
the `"probit"` predictive for classification.

> [!IMPORTANT]
> Laplace expects all data loaders, e.g. `train_loader` and `val_loader` below,
> to be instances of PyTorch
> [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
> Each batch, `next(iter(data_loader))` must either be the standard `(X, y)` tensors
> or a dict-like object containing at least the keys specified in
> `dict_key_x` and `dict_key_y` in Laplace's constructor.

> [!IMPORTANT]
> The total number of data points in all data loaders must be accessible via
> `len(train_loader.dataset)`.

> [!IMPORTANT]
> In `optimize_prior_precision`, make sure to match the arguments with
> the ones you want to pass in `la(x, ...)` during prediction.

```python
from laplace import Laplace

# Pre-trained model
model = load_map_model()

# User-specified LA flavor
la = Laplace(model, "classification",
             subset_of_weights="all",
             hessian_structure="diag")
la.fit(train_loader)
la.optimize_prior_precision(
    method="gridsearch",
    pred_type="glm",
    link_approx="probit",
    val_loader=val_loader
)

# User-specified predictive approx.
pred = la(x, pred_type="glm", link_approx="probit")
```

### Marginal likelihood

The marginal likelihood can be used for model selection [10] and is differentiable
for continuous hyperparameters like the prior precision or observation noise.
Here, we fit the library default, KFAC last-layer LA and differentiate
the log marginal likelihood.

```python
from laplace import Laplace

# Un- or pre-trained model
model = load_model()

# Default to recommended last-layer KFAC LA:
la = Laplace(model, likelihood="regression")
la.fit(train_loader)

# ML w.r.t. prior precision and observation noise
ml = la.log_marginal_likelihood(prior_prec, obs_noise)
ml.backward()
```

### Laplace on LLM

> [!TIP]
> This library also supports Huggingface models and parameter-efficient fine-tuning.
> See `examples/huggingface_examples.py` and `examples/huggingface_examples.md`
> for the full exposition.

First, we need to wrap the pretrained model so that the `forward` method takes a
dict-like input. Note that when you iterate over a Huggingface dataloader,
this is what you get by default. Having a dict-like input is nice since different models
have different number of inputs (e.g. GPT-like LLMs only take `input_ids`, while BERT-like
ones take both `input_ids` and `attention_mask`, etc.). Inside this `forward` method you
can do your usual preprocessing like moving the tensor inputs into the correct device.

```python
class MyGPT2(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained("gpt2")
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", config=config
        )

    def forward(self, data: MutableMapping) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits
```

Then you can "select" which parameters of the LLM you want to apply the Laplace approximation
on, by switching off the gradients of the "unneeded" parameters.
For example, we can replicate a last-layer Laplace: (in actual practice, use `Laplace(..., subset_of_weights='last_layer', ...)` instead, though!)

```python
model = MyGPT2(tokenizer)
model.eval()

# Enable grad only for the last layer
for p in model.hf_model.parameters():
    p.requires_grad = False
for p in model.hf_model.score.parameters():
    p.requires_grad = True

la = Laplace(
    model,
    likelihood="classification",
    # Will only hit the last-layer since it's the only one that is grad-enabled
    subset_of_weights="all",
    hessian_structure="diag",
)
la.fit(dataloader)
la.optimize_prior_precision()

test_data = next(iter(dataloader))
pred = la(test_data)
```

This is useful because we can apply the LA only on the parameter-efficient finetuning
weights. E.g., we can fix the LLM itself, and apply the Laplace approximation only
on the LoRA weights. Huggingface will automatically switch off the non-LoRA weights'
gradients.

```python
def get_lora_model():
    model = MyGPT2(tokenizer)  # Note we don't disable grad
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],  # LoRA on the attention weights
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    return lora_model

lora_model = get_lora_model()

# Train it as usual here...

lora_model.eval()

lora_la = Laplace(
    lora_model,
    likelihood="classification",
    subset_of_weights="all",
    hessian_structure="diag",
    backend=AsdlGGN,
)

test_data = next(iter(dataloader))
lora_pred = lora_la(test_data)
```

### Subnetwork Laplace

This example shows how to fit the Laplace approximation over only
a subnetwork within a neural network (while keeping all other parameters
fixed at their MAP estimates), as proposed in [11]. It also exemplifies
different ways to specify the subnetwork to perform inference over.

First, we make use of `SubnetLaplace`, where we specify the subnetwork by
generating a list of indices for the active model parameters.

```python
from laplace import Laplace

# Pre-trained model
model = load_model()

# Examples of different ways to specify the subnetwork
# via indices of the vectorized model parameters
#
# Example 1: select the 128 parameters with the largest magnitude
from laplace.utils import LargestMagnitudeSubnetMask
subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=128)
subnetwork_indices = subnetwork_mask.select()

# Example 2: specify the layers that define the subnetwork
from laplace.utils import ModuleNameSubnetMask
subnetwork_mask = ModuleNameSubnetMask(model, module_names=["layer.1", "layer.3"])
subnetwork_mask.select()
subnetwork_indices = subnetwork_mask.indices

# Example 3: manually define the subnetwork via custom subnetwork indices
import torch
subnetwork_indices = torch.tensor([0, 4, 11, 42, 123, 2021])

# Define and fit subnetwork LA using the specified subnetwork indices
la = Laplace(model, "classification",
             subset_of_weights="subnetwork",
             hessian_structure="full",
             subnetwork_indices=subnetwork_indices)
la.fit(train_loader)
```

Besides `SubnetLaplace`, you can, as already mentioned, also treat the last
layer only using `Laplace(..., subset_of_weights='last_layer')`, which uses
`LLLaplace`. As a third method, you may define a subnetwork by disabling
gradients of fixed model parameters. The different methods target different use
cases. Each method has pros and cons, please see [this
discussion](https://github.com/aleximmer/Laplace/issues/217#issuecomment-2278311460)
for details. In summary

- Disable-grad: General method to perform Laplace on specific types of
  layer/parameter, e.g. in an LLM with LoRA. Can be used to emulate `LLLaplace`
  as well. Always use `subset_of_weights='all'` for this method.
  - subnet selection by disabling grads is more efficient than
    `SubnetLaplace` since it avoids calculating full Jacobians first
  - disabling grads can only be performed on `Parameter` level and not for
    individual weights, so this doesn't cover all cases that `SubnetLaplace`
    offers such as `Largest*SubnetMask` or `RandomSubnetMask`
- `LLLaplace`: last-layer specific code with improved performance (#145)
- `SubnetLaplace`: more fine-grained partitioning such as
  `LargestMagnitudeSubnetMask`

### Serialization

As with plain `torch`, we support to ways to serialize data.

One is the familiar `state_dict` approach. Here you need to save and re-create
both `model` and `Laplace`. Use this for long-term storage of models and
sharing of a fitted `Laplace` instance.

```py
# Save model and Laplace instance
torch.save(model.state_dict(), "model_state_dict.bin")
torch.save(la.state_dict(), "la_state_dict.bin")

# Load serialized data
model2 = MyModel(...)
model2.load_state_dict(torch.load("model_state_dict.bin"))
la2 = Laplace(model2, "classification",
              subset_of_weights="all",
              hessian_structure="diag")
la2.load_state_dict(torch.load("la_state_dict.bin"))
```

The second approach is to save the whole `Laplace` object, including
`self.model`. This is less verbose and more convenient since you have the
trained model and the fitted `Laplace` data stored in one place, but [also comes with
some
drawbacks](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference).
Use this for quick save-load cycles during experiments, say.

```py
# Save Laplace, including la.model
torch.save(la, "la.pt")

# Load both
torch.load("la.pt")
```

Some Laplace variants such as `LLLaplace` might have trouble being serialized
using the default `pickle` module, which `torch.save()` and `torch.load()` use
(`AttributeError: Can't pickle local object ...`). In this case, the
[`dill`](https://github.com/uqfoundation/dill) package will come in handy.

```py
import dill

torch.save(la, "la.pt", pickle_module=dill)
```

With both methods, you are free to switch devices, for instance when you
trained on a GPU but want to run predictions on CPU. In this case, use

```py
torch.load(..., map_location="cpu")
```

> [!WARNING]
> Currently, this library always assumes that the model has an
> output tensor of shape `(batch_size, ..., n_classes)`, so in
> the case of image outputs, you need to rearrange from NCHW to NHWC.

## Structure

The laplace package consists of two main components:

1. The subclasses of [`laplace.BaseLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/baselaplace.py) that implement different sparsity structures: different subsets of weights (`'all'`, `'subnetwork'` and `'last_layer'`) and different structures of the Hessian approximation (`'full'`, `'kron'`, `'lowrank'`, `'diag'` and `'gp'`). This results in _ten_ currently available options: `laplace.FullLaplace`, `laplace.KronLaplace`, `laplace.DiagLaplace`, `laplace.FunctionalLaplace` the corresponding last-layer variations `laplace.FullLLLaplace`, `laplace.KronLLLaplace`, `laplace.DiagLLLaplace` and `laplace.FunctionalLLLaplace` (which are all subclasses of [`laplace.LLLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/lllaplace.py)), [`laplace.SubnetLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/subnetlaplace.py) (which only supports `'full'` and `'diag'` Hessian approximations) and `laplace.LowRankLaplace` (which only supports inference over `'all'` weights). All of these can be conveniently accessed via the [`laplace.Laplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/laplace.py) function.
2. The backends in [`laplace.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/) which provide access to Hessian approximations of
   the corresponding sparsity structures, for example, the diagonal GGN.

Additionally, the package provides utilities for
decomposing a neural network into feature extractor and last layer for `LLLaplace` subclasses ([`laplace.utils.feature_extractor`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/feature_extractor.py))
and
effectively dealing with Kronecker factors ([`laplace.utils.matrix`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/matrix.py)).

Finally, the package implements several options to select/specify a subnetwork for `SubnetLaplace` (as subclasses of [`laplace.utils.subnetmask.SubnetMask`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/subnetmask.py)).
Automatic subnetwork selection strategies include: uniformly at random (`laplace.utils.subnetmask.RandomSubnetMask`), by largest parameter magnitudes (`LargestMagnitudeSubnetMask`), and by largest marginal parameter variances (`LargestVarianceDiagLaplaceSubnetMask` and `LargestVarianceSWAGSubnetMask`).
In addition to that, subnetworks can also be specified manually, by listing the names of either the model parameters (`ParamNameSubnetMask`) or modules (`ModuleNameSubnetMask`) to perform Laplace inference over.

## Extendability

To extend the laplace package, new `BaseLaplace` subclasses can be designed, for example,
Laplace with a block-diagonal Hessian structure.
One can also implement custom subnetwork selection strategies as new subclasses of `SubnetMask`.

Alternatively, extending or integrating backends (subclasses of [`curvature.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvature.py)) allows to provide different Hessian
approximations to the Laplace approximations.
For example, currently the [`curvature.CurvlinopsInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvlinops.py) based on [Curvlinops](https://github.com/f-dangel/curvlinops) and the native `torch.func` (previously known as `functorch`), [`curvature.BackPackInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/backpack.py) based on [BackPACK](https://github.com/f-dangel/backpack/) and [`curvature.AsdlInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/asdl.py) based on [ASDL](https://github.com/kazukiosawa/asdfghjkl) are available.

## When to use which backend

> [!TIP]
> Each backend as its own caveat/behavior. The use the following to guide you
> picking the suitable backend, depending on you model & application.

- **Small, simple MLP, or last-layer Laplace:** Any backend should work well.
  `CurvlinopsGGN` or `CurvlinopsEF` is recommended if
  `hessian_factorization = 'kron'`, but it's inefficient for other factorizations.
- **LLMs with PEFT (e.g. LoRA):** `AsdlGGN` and `AsdlEF` are recommended.
- **Continuous Bayesian optimization:** `CurvlinopsGGN/EF` and `BackpackGGN/EF` are
  recommended since they are the only ones supporting backprop over Jacobians.

> [!CAUTION]
> The `curvlinops` backends are inefficient for full and diagonal factorizations.
> Moreover, they're also inefficient for computing the Jacobians of large models
> since they rely on `torch.func.jacrev` along `torch.func.vmap`!
> Finally, `curvlinops` only computes K-FAC (`hessian_factorization = 'kron'`)
> for `nn.Linear` and `nn.Conv2d` modules (including those inside larger modules
> like Attention).

> [!CAUTION]
> The `BackPack` backends are limited to models expressed as `nn.Sequential`.
> Also, they're not compatible with normalization layers.

## Documentation

The documentation is available [here](https://aleximmer.github.io/Laplace) or can be generated and/or viewed locally:

**With `uv`**

```bash
# assuming the repository was cloned
uv sync --all-extras
# create docs and write to html
uv run bash update_docs.sh
# .. or serve the docs directly
uv run pdoc --http 0.0.0.0:8080 laplace --template-dir template
```

**With `pip`**

```bash
# assuming the repository was cloned
pip install -e ".[dev]"
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```

## Contributing

Pull requests are very welcome. Please follow these guidelines:

1. Follow the [development setup](#setup-dev-environment).
2. Use [ruff](https://github.com/astral-sh/ruff) as autoformatter. Please refer to the following [makefile](https://github.com/aleximmer/Laplace/blob/main/makefile) and run it via `make ruff`. Please note that the order of `ruff check --fix` and `ruff format` is important!
3. Also use [ruff](https://github.com/astral-sh/ruff) as linter. Please manually fix all linting errors/warnings before opening a pull request.
4. Fully document your changes in the form of Python docstrings, typehinting, and (if applicable) code/markdown examples in the `./examples` subdirectory.
5. Provide as many test cases as possible. Make sure all test cases pass.

Issues, bug reports, and ideas are also very welcome!

## Useful links

- Publishing package with `uv`: <https://docs.astral.sh/uv/guides/publish/>

## References

This package relies on various improvements to the Laplace approximation for neural networks, which was originally due to MacKay [1]. Please consider citing the respective papers if you use any of their proposed methods via our laplace library.

- [1] MacKay, DJC. [_A Practical Bayesian Framework for Backpropagation Networks_](https://authors.library.caltech.edu/13793/). Neural Computation 1992.
- [2] Gibbs, M. N. [_Bayesian Gaussian Processes for Regression and Classification_](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.1130&rep=rep1&type=pdf). PhD Thesis 1997.
- [3] Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N., Sundaram, N., Patwary, M., Prabhat, M., Adams, R. [_Scalable Bayesian Optimization Using Deep Neural Networks_](https://arxiv.org/abs/1502.05700). ICML 2015.
- [4] Ritter, H., Botev, A., Barber, D. [_A Scalable Laplace Approximation for Neural Networks_](https://openreview.net/forum?id=Skdvd2xAZ). ICLR 2018.
- [5] Foong, A. Y., Li, Y., Hernández-Lobato, J. M., Turner, R. E. [_'In-Between' Uncertainty in Bayesian Neural Networks_](https://arxiv.org/abs/1906.11537). ICML UDL Workshop 2019.
- [6] Khan, M. E., Immer, A., Abedi, E., Korzepa, M. [_Approximate Inference Turns Deep Networks into Gaussian Processes_](https://arxiv.org/abs/1906.01930). NeurIPS 2019.
- [7] Kristiadi, A., Hein, M., Hennig, P. [_Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks_](https://arxiv.org/abs/2002.10118). ICML 2020.
- [8] Immer, A., Korzepa, M., Bauer, M. [_Improving predictions of Bayesian neural nets via local linearization_](https://arxiv.org/abs/2008.08400). AISTATS 2021.
- [9] Sharma, A., Azizan, N., Pavone, M. [_Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Networks_](https://arxiv.org/abs/2102.12567). UAI 2021.
- [10] Immer, A., Bauer, M., Fortuin, V., Rätsch, G., Khan, EM. [_Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning_](https://arxiv.org/abs/2104.04975). ICML 2021.
- [11] Daxberger, E., Nalisnick, E., Allingham, JU., Antorán, J., Hernández-Lobato, JM. [_Bayesian Deep Learning via Subnetwork Inference_](https://arxiv.org/abs/2010.14689). ICML 2021.
