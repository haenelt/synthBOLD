- make comments, docstrings, tests, docs
    - __main__
    - base
    - config
    - decorator
    - labels
    - utils

- make Ellipsoid
- make Toroid
- make Triangles
- make Squares
- example script: make gifs of saved labels

- add transforms
- add models

- [ ] how to mix Vessels with other shapes
- [ ] add TE dimension
- [ ] add temporal dimension: BOLD time courses (hrf, autocorrelation, global signal changes, noise)
- [ ] real and complex data
- [ ] augment phase data as well

- [ ] not only constant susceptibility within each vessel but also smooth variation
- [ ] include further transformations
    - gamma transform (randomize gamma exponent)
    - random non-central chi noise
    - multiplicative gamma noise
    - perlin-like noise
    - random localized spikes in k-space
    - random adjust image contrast
    - rician noise
    - gibbs noise
    - gaussian sharpening
    - randomly transform intensity histograms
- [ ] normalization
    - map the 1st and 99th percentiles to 0 and 1, respectively

- [ ] publish docs
- [ ] publish docker
- [ ] zenodo
- [ ] pypi
