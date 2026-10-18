# Scenario-Actor — interactive paper

A minimal, white-first visualization of **Read the Room Before Your Model Generates** by Muyao Kong and Zhihao Lin.

- Website: https://muyao-kong.github.io/scenario-actor/
- Compare styles: https://muyao-kong.github.io/scenario-actor/compare/
- Previous version (A): https://muyao-kong.github.io/scenario-actor/compare/original/
- Simplified version (B): https://muyao-kong.github.io/scenario-actor/compare/simplified/
- Paper: https://arxiv.org/abs/2610.18820
- Share: [vector QR](assets/scenario-actor-qr.svg) · [1600px PNG](assets/scenario-actor-qr.png)

## Explore

The paper is presented as four full-screen chapters: Overview, Method, Experiments, and Paper. Use the centered chapter text and down arrow at the bottom to continue, or the previous-chapter text and up arrow on its left to return one chapter. The top navigation also provides direct access. Smooth vertical transitions and lightly staggered content entrances with a 40px directional offset follow the direction of travel, including content revealed while scrolling within a chapter. Hovering or keyboard-focusing the navigation enlarges its text by 10% and gives the arrow two directional nudges of 7px and 3px. The layout has no enclosing card borders, filled navigation buttons, or boxed control panels. Only the current chapter is interactive. Long content scrolls within its chapter, while the next-chapter link stays visible. Browser back/forward and direct chapter links work; simulation parameters, camera orientation, and progress persist when revisiting Method. Playback pauses when leaving it. Reduced-motion preferences skip transitions, content entrances, and hover motion. Without JavaScript the content remains a conventional readable document.

The introductory 3D scene rotates automatically. Drag it (or use arrow keys) to adjust the view; automatic rotation resumes smoothly after two idle seconds. Automatic motion respects reduced-motion preferences.

**Scenarize:** rotate the 3D view; scrub or play individual optimization steps; adjust the margin and scenario; switch to recurrent prompt inference (Equation 1). Embeddings, centroids, squared triplet distances, and loss update together. The teaching RNN uses real backpropagation through time on synthetic inputs.

**Actout:** send the current embedding into the next module, drag it, rotate it with playback or arrow keys, inspect five axes, and change endpoint temperatures. Signed cosine similarity, normalized weights, temperature, and illustrative token probabilities follow Equations 3 and the temperature softmax.

**Experiments:** compare the three main benchmarks in a compact table and review the geometry ablation and task-only boundary.

## Style comparison

The main page uses the selected split introduction, original SVG logo, enlarged author/date text, and the second version’s interface. Actout displays Equation 3 with its cosine similarity expanded inline; the main formula, axis definition, and nonzero-denominator condition share one line, separated by commas, and the resource section uses the original open two-column layout with larger, standalone PNG QR download. The archived B version is in `compare/simplified/`. `compare/original/` preserves the earlier version from commit `d17e9c8`, with relative asset paths adjusted and indexing disabled. Both archived versions share their original 2D numerical model in `compare/geometry-2d.js`. The current main page combines the first version’s split introduction with the second version’s controls and results, using a 3D teaching model. The comparison page includes same-width browser screenshots and direct links to both interactive demos. The main URL and QR code are unchanged.

## Run

No build or runtime CDN is needed. With Python 3:

```sh
python3 -m http.server 4173 --bind 127.0.0.1
```

Open http://127.0.0.1:4173/. Math fonts and KaTeX are vendored. To regenerate math/QR assets or run numerical tests:

```sh
npm ci
npm run assets
npm test
```

## Structure

- `index.html`: crawlable content, controls, metadata, and paper links.
- `style.css`: responsive layout, consistent drawing styles, and motion preferences.
- `geometry.js`: small recurrent model, training, paper equations, scenario pairs, reported benchmark values.
- `app.js`: projected 3D SVG drawing, smooth simulation playback, and interactions.
- `deck.js`: chapter navigation, transitions, browser history, and accessible focus management.
- `scene3d.js`: orthographic camera, ray/sphere dragging, spatial angle arcs, and 3D direction controls.
- `assets/`: local math renderer/fonts, QR codes, favicon, and sharing image.
- `scripts/assets.mjs`: reproducible vendor and QR generation.
- `tests/geometry.test.mjs`: numerical properties of the equations and training.

## Scientific scope

The paper uses frozen Llama-3.1-8B-Instruct layer-16 hidden states (4096D), 32D scenario embeddings, nine labels, five axes, and triplet margin 0.3. The website uses a deterministic **3D teaching RNN with four synthetic input features and four prompt chunks**. It is not a PCA/UMAP of the published model, and is not a reproduction of the benchmark experiments. Its optimizer is simple gradient descent; the paper uses AdamW. Changing the margin restarts the teaching run. The teaching model uses tanh for the activation function and a fixed learning rate of 0.0035. Synthetic scenario inputs vary independently in all three coordinates to avoid an almost planar teaching space; the introduction shows three opposing scenario pairs. All rendered points and axes come from the same model, without display-only coordinate offsets.

Actout computes **cosine similarity to centroid-difference axis directions**, rather than similarity to individual raw centroids. It interpolates the two endpoint temperatures and normalizes axis contributions by absolute cosine similarity. Axes are not assumed orthogonal. The normalized point on the unit sphere illustrates the cosine calculation. The angle arc and orthogonal projection are computed in 3D before rendering. Drag empty space (or use the arrow keys) to rotate the orthographic camera; this changes only the view. Drag the Actout embedding on its sphere, or use left/right for azimuth and up/down for elevation, to change its actual 3D direction. Vector magnitude does not affect temperature.

The non-adversarial centroid is the mean of the other eight class centroids, as stated in the paper. Table 2 does not give a separate non-adversarial endpoint temperature. This demo uses the other eight endpoint temperatures' mean (0.875), explicitly marked in the interface. This is an illustrative convention, not a recovered experimental configuration.

All benchmark numbers are transcribed from Tables 1, 3, and 6 of the supplied paper. Statistical context follows Appendix B. In particular, the differences from fixed T=0.7 are not significant on AgentSense and MT-Bench; AlpacaEval improves significantly. The paper's results are established for one model and taxonomy.

## GitHub Pages and search

In repository **Settings → Pages**, choose **Deploy from a branch → main → /(root)**. The site is static and `.nojekyll` disables Jekyll. Subsequent changes to `main` publish automatically.

The canonical URL, description, Open Graph image, scholarly metadata, structured data, sitemap, semantic static content, and index/follow directive are included. Search engines determine whether and when to index the page. For explicit submission, add the URL or sitemap in the owner's Google Search Console/Bing Webmaster Tools. A project-level `robots.txt` does not replace the host-root robots policy.

The QR encodes the direct HTTPS Pages URL with no shortener, tracking, subscription, or expiry mechanism. It remains usable while the repository name, account name, path, and Pages hosting remain available. Regenerate the QR and metadata if that address changes.
