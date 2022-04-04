# csips
**C**arl's **S**imple **I**nteger **P**rogramming **S**olver.

A pure-python integer programming solver using branch-and-bound with scipy's `linprog` for subproblem LP relaxation.

- [x] Integer linear programming problems
- [x] Embedded python DSL: no need to specify standard form
- [x] Pure python: only python dependencies, no need to install other solvers

**CSIPS does not support**

- [ ] Non-integer variables (real-valued)
- [ ] Non-linear constraints (quadratic or otherwise)
- [ ] Input/output formats compatible with other solvers.

## Using

Add csips as a submodule for now, then use like `from csips import csips`.
You may be able to aim your package manager at this git repo and use it that way as well.
Then you should only ened `import csips`

## Specifying your problem

...

## Solving your problem

...

## Tests
```
poetry run pytest
```

## Contributing

This was written purely as an educational exercise.

1. Install poetry.
2. Use poetry to install project dependencies.
3. Make sure vscode python interpreter is the poetry venv one.

## License

This project is available under the terms of GPL 3.0. 
Please see [LICENSE](LICENSE) for details.