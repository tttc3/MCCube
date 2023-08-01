# How to Contribute to MCCube
Any contributions are very welcome, and are received with great gratitude.

*Once the required CI/CD pipelines have been established, pull requests will be very 
much welcomed, and greatly appreciated. These should be ready very soon, but in the 
meantime, please feel free to open a draft pull request.*

## Enhancing code
To get started you will need to fork MCCube on GitHub, then install it in editable mode':

```bash
git clone https://github.com/your-username-here/mccube
cd mccube
pip install -e ".[dev]"
```

You will then want to install the pre-commit hooks, that will ensure your commits pass 
expected formatting and linting rules (black and ruff respectively):

```bash
pre-commit install
```

You can now go ahead and make enhancements to your fork of MCCube.

### Testing your enhancements
Once you are happy with your enhancements it is important to check that all the existing 
tests, and any new tests you may have implemented (to test the functionality of your 
enhancements), pass successfully. To do this, run:

```bash
pytest
```

### Documenting your enhancements
If all the test pass, the final thing to do is document your enhancements. It is a good 
idea to check that any changes you make to the documentation render correctly when it
is built with Sphinx. To check this, run the following command, and then browse from 
`docs/_build/html/index.html`:

```bash
pip install -r docs/requirements-docs.txt
sphinx-build -b html docs docs/_build/html -j auto
```

## Submitting your pull request
Once you are satisfied that your enhancements pass the required tests and are well 
documented, push your committed code with `git push` and open a pull request 
on GitHub (instructions can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).

