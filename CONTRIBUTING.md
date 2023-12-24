# How to Contribute to MCCube
Any contributions are very welcome and are received with great gratitude.

## Enhancing code
To get started you will need to fork MCCube on GitHub, then install it in editable mode':

```bash
git clone https://github.com/your-username-here/mccube
cd mccube
pip install -e ".[dev]"
```

You will then want to install the pre-commit hooks. These will ensure your commits pass 
expected formatting, linting and type hinting rules (ruff and pyright respectively):

```bash
pre-commit install
```

You can now go ahead and make enhancements to your fork of MCCube.

### Testing your enhancements
Once you are happy with your enhancements, it is important to check that all the existing 
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

Alternatively you can run

```bash
pip install -r docs/requirements-docs.txt
sphinx-autobuild docs docs/_build/html
```

which will start a server at [http://127.0.0.1:8000](http://127.0.0.1:8000), from where
you can view the documentation. This method has the added bonus of automatically 
rebuilding the documentation upon detecting any change to the files in the `docs` folder.

## Submitting your pull request
Once you are satisfied that your enhancements pass the required tests and are well 
documented, push your committed code with `git push` and open a pull request 
on GitHub (see instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).

