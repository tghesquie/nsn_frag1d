# Third-Party Dependencies

This project builds and links against the following third-party software.

## Akantu

- **Repository:** https://gitlab.com/akantu/akantu.git
- **Pinned commit:** `22adc1e143ca74fdb70af185536d16ff4a3396de`
- **License:** GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)
- **Usage:** Finite-element engine with cohesive elements and contact mechanics,
  accessed via the Python interface compiled during `workflows/install.sh`.

The Akantu source code is cloned into `external/akantu/` by `workflows/install.sh` and is
not part of this repository. Its license terms apply to that cloned copy.
