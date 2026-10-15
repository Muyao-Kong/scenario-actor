# Scenario-Actor — interactive paper

A minimal, white-first visualization of **Read the Room Before Your Model Generates** by Muyao Kong and Zhihao Lin.

- Website: https://muyao-kong.github.io/scenario-actor/
- Paper: https://arxiv.org/abs/2610.18820
- Share: [vector QR](assets/scenario-actor-qr.svg) · [1600px PNG](assets/scenario-actor-qr.png)

## Explore

**Scenarize:** scrub or play individual optimization steps; adjust margin, learning rate, and scenario; switch to recurrent prompt inference (Figure 2 / Equation 1). Embeddings, centroids, squared triplet distances, and loss update together. The teaching RNN uses real backpropagation through time on synthetic inputs.

**Actout:** send the current embedding into the next module, drag it, rotate it with the direction slider or arrow keys, inspect five axes, and change endpoint temperatures. Signed cosine similarity, normalized weights, temperature, and illustrative token probabilities follow Equations 3 and the temperature softmax.

**Experiments:** switch among the three main benchmarks, open the complete results table, and review the geometry ablation and task-only boundary.

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
- `app.js`: SVG drawing, smooth simulation playback, and interactions.
- `assets/`: local math renderer/fonts, QR codes, favicon, and sharing image.
- `scripts/assets.mjs`: reproducible vendor and QR generation.
- `tests/geometry.test.mjs`: numerical properties of the equations and training.

## Scientific scope

The paper uses frozen Llama-3.1-8B-Instruct layer-16 hidden states (4096D), 32D scenario embeddings, nine labels, five axes, and triplet margin 0.3. The website uses a deterministic **2D teaching RNN with four synthetic input features and four prompt chunks**. It is not a PCA/UMAP of the published model, and is not a reproduction of the benchmark experiments. Its optimizer is simple gradient descent; the paper uses AdamW. Small learning-rate changes restart the teaching run.

Actout computes **cosine similarity to centroid-difference axis directions**, rather than similarity to individual raw centroids. It interpolates the two endpoint temperatures and normalizes axis contributions by absolute cosine similarity. Axes are not assumed orthogonal. The normalized point on the unit circle illustrates the cosine calculation. Vector magnitude does not affect temperature.

The non-adversarial centroid is the mean of the other eight class centroids, as stated in the paper. Table 2 does not give a separate non-adversarial endpoint temperature. This demo uses the other eight endpoint temperatures' mean (0.875), explicitly marked in the interface. This is an illustrative convention, not a recovered experimental configuration.

All benchmark numbers are transcribed from Tables 1, 3, and 6 of the supplied paper. Statistical context follows Appendix B. In particular, the differences from fixed T=0.7 are not significant on AgentSense and MT-Bench; AlpacaEval improves significantly. The paper's results are established for one model and taxonomy.

## GitHub Pages and search

In repository **Settings → Pages**, choose **Deploy from a branch → main → /(root)**. The site is static and `.nojekyll` disables Jekyll. Subsequent changes to `main` publish automatically.

The canonical URL, description, Open Graph image, scholarly metadata, structured data, sitemap, semantic static content, and index/follow directive are included. Search engines determine whether and when to index the page. For explicit submission, add the URL or sitemap in the owner's Google Search Console/Bing Webmaster Tools. A project-level `robots.txt` does not replace the host-root robots policy.

The QR encodes the direct HTTPS Pages URL with no shortener, tracking, subscription, or expiry mechanism. It remains usable while the repository name, account name, path, and Pages hosting remain available. Regenerate the QR and metadata if that address changes.
