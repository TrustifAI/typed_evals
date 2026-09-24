# Publishing to PyPI

The release workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and generates distribution attestations. It does not use a stored API token.

## One-time setup

Create a GitHub environment named `pypi` in `TrustifAI/typed_evals`. Restrict its
deployment tags to `v*` and add required reviewers if available for the repository.

On PyPI, configure a GitHub Trusted Publisher for the following values:

| Setting | Value |
| --- | --- |
| PyPI project | `typed_evals` |
| GitHub owner | `TrustifAI` |
| Repository | `typed_evals` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

For a new project, register a pending publisher under your
[PyPI account publishing settings](https://pypi.org/manage/account/publishing/).
For an existing project, add the publisher under that project's publishing settings.
These settings must be configured before the first release using this workflow.
After confirming Trusted Publishing works, revoke the old PyPI API token and remove
the unused `PYPI_TOKEN` GitHub secret if no other workflow uses it.

## Release a version

1. Update `version` in `pyproject.toml` and `__version__` in
   `typed_evals/__init__.py` to the same new version.
2. Commit and push the release changes, including both workflow files.
3. Create and push a matching tag, for example `v0.1.0` for version `0.1.0`:

   ```sh
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```

The workflow checks the tag against the package version, runs the Python
3.11–3.13 test matrix and lint checks, builds a wheel from the source distribution,
checks package metadata, and smoke tests the installed wheel. Publishing waits for
all checks to succeed and for any configured environment approval.

Use a version that has not already been published. Duplicate uploads fail;
the workflow does not silently skip existing distributions.

A manual run on a branch performs validation without publishing. A manual run
targeting a matching version tag can publish, for example:

```sh
gh workflow run publish.yml --ref v0.1.0
```

GitHub Actions are pinned to verified release commits. Dependabot proposes weekly
updates to those pins for review.
