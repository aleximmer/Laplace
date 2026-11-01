The Curvlinops backends are the default and are always installed. The ASDL and
BackPACK backends need the `asdl` and `backpack` extras, e.g.
`pip install "laplace-torch[asdl,backpack]"`.

::: laplace.curvature
    options:
        members:
            - CurvatureInterface
            - GGNInterface
            - EFInterface
            - CurvlinopsInterface
            - CurvlinopsGGN
            - CurvlinopsEF
            - CurvlinopsHessian

::: laplace.curvature.asdl
    options:
        members:
            - AsdlInterface
            - AsdlGGN
            - AsdlEF
            - AsdlHessian

::: laplace.curvature.backpack
    options:
        members:
            - BackPackInterface
            - BackPackGGN
            - BackPackEF
