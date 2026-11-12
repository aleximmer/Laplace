# Likelihoods

::: laplace.Likelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.GaussianLikelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.CategoricalLikelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.BernoulliLikelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.BradleyTerryLikelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.HeteroscedasticGaussianLikelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"

::: laplace.get_likelihood
    options:
        filters:
            - "!^_[^_]"
            - "!^(REGRESSION|CLASSIFICATION|REWARD_MODELING|BERNOULLI)$"
            - "!^__(eq|hash|str|repr|init_subclass)__$"
