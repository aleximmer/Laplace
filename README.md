<div align="center">
	<img src="https://raw.githubusercontent.com/AlexImmer/Laplace/main/logo/laplace_logo.png" alt="Laplace" width="300"/>
</div>

[![Main](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=main)](https://travis-ci.com/AlexImmer/Laplace)


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

## Setup

For full compatibility, install this package in a fresh virtual env.
We assume Python >= 3.9 since lower versions are [(soon to be) deprecated](https://devguide.python.org/versions/).
PyTorch version 2.0 and up is also required for full compatibility.
To install laplace with `pip`, run the following:

```bash
pip install laplace-torch
```

For development purposes, clone the repository and then install:

```bash
# or after cloning the repository for development
pip install -e .
# run tests
pip install -e '.[tests]'
pytest tests/
```

## Example usage

### _Post-hoc_ prior precision tuning of diagonal LA

In the following example, a pre-trained model is loaded,
then the Laplace approximation is fit to the training data
(using a diagonal Hessian approximation over all parameters),
and the prior precision is optimized with cross-validation `"gridsearch"`.
After that, the resulting LA is used for prediction with
the `"probit"` predictive for classification.

```python
from laplace import Laplace

# Pre-trained model
model = load_map_model()

# User-specified LA flavor
la = Laplace(model, "classification",
             subset_of_weights="all",
             hessian_structure="diag")
la.fit(train_loader)
la.optimize_prior_precision(method="gridsearch", val_loader=val_loader)

# User-specified predictive approx.
pred = la(x, link_approx="probit")
```

### Differentiating the log marginal likelihood w.r.t. hyperparameters

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

### Laplace on foundation models like LLMs

This library also supports Huggingface models and parameter-efficient fine-tuning.
See `examples/huggingface_examples.py` for a runnable example.

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
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits

# Last-layer Laplace on the foundation model itself
# -------------------------------------------------
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

pred = la(next(iter(dataloader))

# Laplace on the LoRA-attached LLM
# --------------------------------
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

lora_pred = lora_la(next(iter(dataloader)))
```

### Applying the LA over only a subset of the model parameters

This example shows how to fit the Laplace approximation over only
a subnetwork within a neural network (while keeping all other parameters
fixed at their MAP estimates), as proposed in [11]. It also exemplifies
different ways to specify the subnetwork to perform inference over.

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

## Structure

The laplace package consists of two main components:

1. The subclasses of [`laplace.BaseLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/baselaplace.py) that implement different sparsity structures: different subsets of weights (`'all'`, `'subnetwork'` and `'last_layer'`) and different structures of the Hessian approximation (`'full'`, `'kron'`, `'lowrank'`, `'diag'` and `'gp'`). This results in _ten_ currently available options: `laplace.FullLaplace`, `laplace.KronLaplace`, `laplace.DiagLaplace`, `laplace.FunctionalLaplace` the corresponding last-layer variations `laplace.FullLLLaplace`, `laplace.KronLLLaplace`,  `laplace.DiagLLLaplace` and `laplace.FunctionalLLLaplace` (which are all subclasses of [`laplace.LLLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/lllaplace.py)), [`laplace.SubnetLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/subnetlaplace.py) (which only supports `'full'` and `'diag'` Hessian approximations) and `laplace.LowRankLaplace` (which only supports inference over `'all'` weights). All of these can be conveniently accessed via the [`laplace.Laplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/laplace.py) function.
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

The `curvature.CurvlinopsInterface` backend is the default and provides all Hessian approximation variants except the low-rank Hessian.
For the latter, `curvature.AsdlInterface` can be used.
Note that `curvature.AsdlInterface` and `curvature.BackPackInterface` are less complete and less compatible than `curvature.CurvlinopsInterface`.
So, we recommend to stick with `curvature.CurvlinopsInterface` unless you have a specific need of ASDL or BackPACK.

## Documentation

The documentation is available [here](https://aleximmer.github.io/Laplace) or can be generated and/or viewed locally:

```bash
# assuming the repository was cloned
pip install -e .[docs]
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```

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
