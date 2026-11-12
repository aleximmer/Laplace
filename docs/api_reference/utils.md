# Utilities

::: laplace.utils
    options:
        # Its own filters, since subnet masks expose `n_params_subnet`, which the
        # global ones hide
        filters:
            - "!^_[^_]"
            - "!^(damping|enable_backprop)$"
        members:
            - "FeatureExtractor"
            - "Kron"
            - "KronDecomposed"
            - "SubnetMask"
            - "RandomSubnetMask"
            - "LargestMagnitudeSubnetMask"
            - "LargestVarianceDiagLaplaceSubnetMask"
            - "LargestVarianceSWAGSubnetMask"
            - "ParamNameSubnetMask"
            - "ModuleNameSubnetMask"
            - "LastLayerSubnetMask"
            - "RunningNLLMetric"
            - "RunningBernoulliNLLMetric"
            - "RunningGaussianNLLMetric"
