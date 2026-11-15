# Vendored libraries

Third-party scripts shipped with the UI, so it runs with no internet access
and no build step. Files are unmodified release builds, pinned by version.

| File | Library | Version | License | Source |
|---|---|---|---|---|
| `marked.min.js` | marked | 12.0.2 | MIT | https://github.com/markedjs/marked |
| `purify.min.js` | DOMPurify | 3.1.6 | Apache-2.0 OR MPL-2.0 | https://github.com/cure53/DOMPurify |

To update: download the same file for the new version from the package's
npm release (`marked.min.js`, `dist/purify.min.js`) and change this table.
