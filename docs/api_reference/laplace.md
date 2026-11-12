# Laplace Frontend

The public API is every name in `laplace.__all__`, `laplace.curvature.__all__`
and `laplace.utils.__all__`, plus the methods and attributes shown on these
pages. Import Laplace classes, likelihoods, enums and `marglik_training` from
`laplace`, curvature backends from `laplace.curvature`, and subnet masks,
metrics, `FeatureExtractor`, `Kron` and `KronDecomposed` from `laplace.utils`.

Submodule paths such as `laplace.baselaplace` or `laplace.utils.enums` are not
public and may move. These pages leave out internal class members, including
some without a leading underscore. A public name is removed or renamed only
after at least one release in which using it raises a `DeprecationWarning`.

::: laplace.Laplace
