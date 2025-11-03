
## VLM-GIST — Vision-Language Model Grounding for Instance Segmentation & Tracking

Official implementation of: [*Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking*](https://vlm-gist.github.io) by Bastian Pätzold, Jan Nogga and [Sven Behnke](https://www.ais.uni-bonn.de/behnke). IEEE Robotics and Automation Letters (RA-L). 2025.

### Benchmark

We provide [detailed results](./notebooks/evaluation/evaluation.ipynb) and [all experimental settings](./settings).

<details>
<summary>COCO minival (subset)</summary>

```bash
Model                        |  F1   |  Rec. | Prec. |  mAP  || Ins. | Mat. |  Time  || Fail | Ret.
---------------------------------------------------------------------------------------------------
Gemini 2.5 Pro (high)        | 0.541 | 0.489 | 0.606 | 0.338 || 11.4 | 0.50 |  14.5s ||  nan |  nan
Gemini 2.5 Pro (preview)     | 0.537 | 0.487 | 0.599 | 0.350 || 11.1 | 0.51 |  13.3s ||  nan |  nan
Gemini 2.5 Pro               | 0.537 | 0.487 | 0.598 | 0.337 || 11.4 | 0.50 |  14.7s ||  nan |  nan
GLM 4.5V (think)             | 0.526 | 0.445 | 0.642 | 0.334 ||  8.2 | 0.60 |  14.9s ||  nan |  nan
Grok 4                       | 0.524 | 0.531 | 0.518 | 0.340 || 12.1 | 0.59 |  58.2s ||  nan |  nan
InternVL 3.5 38B (think)     | 0.523 | 0.464 | 0.599 | 0.336 ||  9.8 | 0.55 |  59.1s ||  nan |  nan
Gemini 2.5 Flash (medium)    | 0.520 | 0.487 | 0.558 | 0.323 || 12.7 | 0.48 |   6.6s ||  nan |  nan
Gemini 2.5 Flash Lite (high) | 0.520 | 0.449 | 0.616 | 0.316 ||  8.9 | 0.57 |   9.5s || 0.00 | 0.05
Gemini 2.5 Flash (low)       | 0.518 | 0.479 | 0.563 | 0.326 || 12.1 | 0.49 |   5.7s ||  nan |  nan
InternVL 3.5 20B A4B (think) | 0.518 | 0.457 | 0.597 | 0.312 ||  9.2 | 0.58 |  40.1s ||  nan |  nan
Gemini 2.5 Flash (high)      | 0.515 | 0.493 | 0.540 | 0.327 || 13.0 | 0.49 |   7.6s ||  nan |  nan
Qwen3-VL 235B A22B (think)   | 0.513 | 0.459 | 0.580 | 0.324 ||  8.5 | 0.66 |   5.0s || 0.00 | 0.14
GLM 4.5V                     | 0.510 | 0.412 | 0.670 | 0.328 ||  7.4 | 0.60 |  15.9s || 0.03 | 0.19
Gemini 2.5 Flash             | 0.509 | 0.453 | 0.579 | 0.321 ||  9.5 | 0.58 |   2.9s ||  nan |  nan
GPT-4.1                      | 0.508 | 0.472 | 0.550 | 0.330 || 10.5 | 0.57 |   7.5s ||  nan |  nan
Gemini 2.5 Flash (preview)   | 0.507 | 0.444 | 0.590 | 0.323 ||  9.2 | 0.57 |   3.2s ||  nan |  nan
GPT-5 mini                   | 0.505 | 0.501 | 0.509 | 0.327 || 13.7 | 0.50 |  27.9s ||  nan |  nan
GPT-4.1 mini                 | 0.504 | 0.426 | 0.618 | 0.297 ||  7.8 | 0.62 |   4.3s ||  nan |  nan
Grok 4 Fast (high)           | 0.504 | 0.485 | 0.525 | 0.313 || 12.7 | 0.52 |   9.1s || 0.01 | 0.51
InternVL 3.5 4B (think)      | 0.501 | 0.432 | 0.597 | 0.292 ||  8.9 | 0.57 |  35.0s ||  nan |  nan
GPT-5 mini (high)            | 0.501 | 0.476 | 0.529 | 0.297 || 13.9 | 0.46 |  82.4s || 0.02 | 0.10
Qwen3-VL 235B A22B           | 0.500 | 0.456 | 0.554 | 0.329 ||  9.4 | 0.62 |   6.1s || 0.01 | 0.10
OVIS 2.5 9B                  | 0.499 | 0.399 | 0.667 | 0.316 ||  7.0 | 0.60 |   6.5s ||  nan |  nan
InternVL 3.5 30B A3B         | 0.499 | 0.410 | 0.635 | 0.306 ||  7.1 | 0.64 |  35.1s ||  nan |  nan
Qwen3-VL 30B A3B (think)     | 0.495 | 0.391 | 0.677 | 0.276 ||  5.8 | 0.70 |  25.4s || 0.01 | 0.09
Grok 4 Fast                  | 0.495 | 0.480 | 0.510 | 0.304 || 13.0 | 0.52 |   9.1s || 0.03 | 0.54
Claude Sonnet 4.5            | 0.491 | 0.394 | 0.650 | 0.294 ||  7.4 | 0.57 |   5.3s || 0.00 | 0.00
InternVL 3.5 30B A3B (think) | 0.490 | 0.415 | 0.600 | 0.289 ||  9.2 | 0.52 |  38.6s ||  nan |  nan
Mistral Medium 3.1 (t=1.00)  | 0.489 | 0.408 | 0.611 | 0.313 ||  8.4 | 0.56 |   2.9s ||  nan |  nan
OVIS 2.5 9B (think)          | 0.489 | 0.394 | 0.644 | 0.301 ||  8.1 | 0.53 |  24.0s ||  nan |  nan
GPT-5 (high)                 | 0.487 | 0.510 | 0.466 | 0.311 || 16.7 | 0.46 |  88.6s ||  nan |  nan
GPT-5 nano (high)            | 0.487 | 0.415 | 0.589 | 0.297 ||  9.4 | 0.53 |  44.2s ||  nan |  nan
GPT-5                        | 0.486 | 0.503 | 0.471 | 0.311 || 15.8 | 0.47 |  47.6s ||  nan |  nan
GPT-5 nano                   | 0.485 | 0.416 | 0.581 | 0.289 ||  9.2 | 0.54 |  24.1s ||  nan |  nan
Claude Sonnet 4.5 (high)     | 0.481 | 0.408 | 0.588 | 0.283 ||  8.4 | 0.58 |  16.6s || 0.00 | 0.01
Gemma 27B                    | 0.478 | 0.418 | 0.558 | 0.284 || 10.6 | 0.50 |   9.5s || 0.00 | 0.00
Claude Sonnet 4              | 0.472 | 0.372 | 0.646 | 0.291 ||  7.9 | 0.51 |   5.5s ||  nan |  nan
Qwen-VL Plus                 | 0.472 | 0.368 | 0.659 | 0.270 ||  6.8 | 0.59 |   2.8s || 0.02 | 0.53
Mistral Medium 3.1 (t=0.15)  | 0.472 | 0.418 | 0.541 | 0.308 ||  9.3 | 0.59 |   3.0s ||  nan |  nan
InternVL 3.5 2B (think)      | 0.470 | 0.388 | 0.595 | 0.294 ||  7.9 | 0.58 |  27.7s ||  nan |  nan
InternVL 3.5 2B              | 0.466 | 0.386 | 0.588 | 0.288 ||  6.8 | 0.67 | 124.8s ||  nan |  nan
Gemma 12B                    | 0.462 | 0.384 | 0.578 | 0.291 ||  9.3 | 0.50 |  12.5s || 0.00 | 0.37
Claude Haiku 4.5 (high)      | 0.461 | 0.396 | 0.551 | 0.258 || 11.1 | 0.45 |  14.7s || 0.00 | 0.00
Claude Haiku 4.5             | 0.459 | 0.379 | 0.582 | 0.256 ||  9.8 | 0.47 |   3.6s || 0.00 | 0.00
Gemini 2.5 Flash Lite        | 0.447 | 0.435 | 0.460 | 0.304 || 10.1 | 0.66 |   2.8s || 0.00 | 0.01
Qwen3-VL 30B A3B             | 0.432 | 0.408 | 0.459 | 0.275 ||  9.6 | 0.67 |   4.4s || 0.03 | 0.50
Gemma 4B                     | 0.430 | 0.323 | 0.641 | 0.255 ||  6.4 | 0.55 |   3.3s || 0.00 | 0.01
GPT-4.1 nano                 | 0.415 | 0.341 | 0.530 | 0.232 ||  6.5 | 0.69 |   3.5s ||  nan |  nan
Qwen3-VL 8B                  | 0.366 | 0.394 | 0.343 | 0.279 ||  9.9 | 0.82 |   6.3s || 0.01 | 0.05
Qwen3-VL 8B (think)          | 0.333 | 0.224 | 0.646 | 0.180 ||  7.0 | 0.61 |  72.0s || 0.43 | 1.64
```

</details>
<details>
<summary>Custom Dataset</summary>

```bash
Model                        |  F1   |  Rec. | Prec. |  mAP  || Ins. | Mat. |  Time  || Fail | Ret.
---------------------------------------------------------------------------------------------------
Gemini 2.5 Pro               | 0.464 | 0.417 | 0.523 | 0.363 || 14.7 | 1.00 |  16.0s ||  nan |  nan
Gemini 2.5 Pro (high)        | 0.460 | 0.415 | 0.515 | 0.366 || 14.8 | 1.00 |  17.7s ||  nan |  nan
Gemini 2.5 Pro (preview)     | 0.451 | 0.402 | 0.515 | 0.357 || 14.3 | 1.00 |  14.9s ||  nan |  nan
Gemini 2.5 Flash (low)       | 0.435 | 0.395 | 0.485 | 0.344 || 15.0 | 1.00 |   7.1s ||  nan |  nan
Grok 4                       | 0.428 | 0.385 | 0.482 | 0.340 || 14.7 | 1.00 |  34.7s ||  nan |  nan
GPT-5 mini (high)            | 0.422 | 0.401 | 0.445 | 0.342 || 16.6 | 1.00 |  82.7s ||  nan |  nan
GPT-5                        | 0.418 | 0.413 | 0.424 | 0.352 || 17.9 | 1.00 |  54.2s ||  nan |  nan
OVIS 2.5 9B (think)          | 0.415 | 0.318 | 0.596 | 0.281 ||  9.8 | 1.00 |  32.0s ||  nan |  nan
Gemini 2.5 Flash (high)      | 0.415 | 0.379 | 0.458 | 0.324 || 15.3 | 1.00 |   7.4s ||  nan |  nan
GPT-5 mini                   | 0.414 | 0.388 | 0.444 | 0.338 || 16.1 | 1.00 |  32.7s ||  nan |  nan
Gemini 2.5 Flash Lite (high) | 0.412 | 0.340 | 0.522 | 0.291 || 12.0 | 1.00 |  11.9s || 0.00 | 0.02
GPT-4.1 mini                 | 0.410 | 0.319 | 0.571 | 0.279 || 10.3 | 1.00 |   6.0s ||  nan |  nan
GPT-4.1                      | 0.407 | 0.356 | 0.476 | 0.308 || 13.8 | 1.00 |   8.7s ||  nan |  nan
Grok 4 Fast (high)           | 0.407 | 0.377 | 0.443 | 0.323 || 15.7 | 1.00 |   9.0s || 0.00 | 0.38
GPT-5 (high)                 | 0.407 | 0.413 | 0.400 | 0.348 || 19.0 | 1.00 | 101.4s ||  nan |  nan
Gemini 2.5 Flash             | 0.406 | 0.345 | 0.495 | 0.305 || 12.8 | 1.00 |   3.6s ||  nan |  nan
InternVL 3.5 38B (think)     | 0.405 | 0.332 | 0.521 | 0.295 || 11.7 | 1.00 |  58.3s ||  nan |  nan
Qwen3-VL 30B A3B (think)     | 0.402 | 0.292 | 0.644 | 0.258 ||  8.3 | 1.00 |  28.9s || 0.00 | 0.16
Gemini 2.5 Flash (preview)   | 0.401 | 0.341 | 0.486 | 0.297 || 12.9 | 1.00 |   4.0s ||  nan |  nan
Gemini 2.5 Flash (medium)    | 0.400 | 0.366 | 0.441 | 0.311 || 15.3 | 1.00 |   7.2s ||  nan |  nan
GLM 4.5V (think)             | 0.400 | 0.312 | 0.555 | 0.276 || 10.4 | 1.00 |  14.9s ||  nan |  nan
Qwen3-VL 235B A22B           | 0.397 | 0.320 | 0.521 | 0.278 || 11.3 | 1.00 |   8.0s || 0.00 | 0.06
Qwen3-VL 235B A22B (think)   | 0.392 | 0.329 | 0.483 | 0.286 || 12.6 | 1.00 |   8.3s || 0.00 | 0.14
Grok 4 Fast                  | 0.387 | 0.357 | 0.424 | 0.309 || 15.5 | 1.00 |   7.8s || 0.00 | 0.16
GPT-5 nano (high)            | 0.383 | 0.319 | 0.478 | 0.280 || 12.3 | 1.00 |  51.3s ||  nan |  nan
InternVL 3.5 30B A3B (think) | 0.378 | 0.302 | 0.506 | 0.259 || 11.0 | 1.00 |  38.5s ||  nan |  nan
GPT-5 nano                   | 0.373 | 0.306 | 0.479 | 0.267 || 11.8 | 1.00 |  28.9s ||  nan |  nan
Qwen3-VL 8B                  | 0.373 | 0.284 | 0.544 | 0.249 ||  9.7 | 1.00 |   4.0s || 0.02 | 0.05
GLM 4.5V                     | 0.368 | 0.279 | 0.538 | 0.243 ||  9.9 | 1.00 |  16.0s || 0.03 | 0.22
OVIS 2.5 9B                  | 0.365 | 0.267 | 0.573 | 0.239 ||  8.6 | 1.00 |   8.0s ||  nan |  nan
InternVL 3.5 4B (think)      | 0.363 | 0.284 | 0.503 | 0.246 || 10.4 | 1.00 |  36.1s ||  nan |  nan
Mistral Medium 3.1 (t=0.15)  | 0.351 | 0.285 | 0.458 | 0.247 || 11.5 | 1.00 |   3.7s ||  nan |  nan
Qwen-VL Plus                 | 0.346 | 0.260 | 0.520 | 0.227 ||  9.2 | 1.00 |   4.4s || 0.00 | 0.34
Gemini 2.5 Flash Lite        | 0.345 | 0.282 | 0.445 | 0.241 || 11.7 | 1.00 |   3.1s || 0.00 | 0.02
InternVL 3.5 30B A3B         | 0.344 | 0.254 | 0.533 | 0.222 ||  8.8 | 1.00 |  42.2s ||  nan |  nan
InternVL 3.5 4B              | 0.343 | 0.262 | 0.495 | 0.229 ||  9.8 | 1.00 |  37.1s ||  nan |  nan
InternVL 3.5 2B (think)      | 0.337 | 0.258 | 0.486 | 0.220 ||  9.8 | 1.00 |  33.1s ||  nan |  nan
Claude Sonnet 4.5 (high)     | 0.334 | 0.279 | 0.414 | 0.233 || 12.4 | 1.00 |  17.5s || 0.00 | 0.00
Qwen3-VL 30B A3B             | 0.330 | 0.244 | 0.512 | 0.215 ||  9.5 | 1.00 |  11.3s || 0.08 | 0.59
Qwen3-VL 8B (think)          | 0.327 | 0.240 | 0.512 | 0.209 || 10.6 | 1.00 |  97.3s || 0.19 | 0.84
Gemma 27B                    | 0.327 | 0.267 | 0.421 | 0.229 || 11.7 | 1.00 |  11.4s || 0.00 | 0.00
Mistral Medium 3.1 (t=1.00)  | 0.327 | 0.284 | 0.383 | 0.247 || 13.7 | 1.00 |   4.0s ||  nan |  nan
Claude Sonnet 4.5            | 0.325 | 0.258 | 0.437 | 0.227 || 10.9 | 1.00 |   7.3s || 0.00 | 0.00
Claude Sonnet 4              | 0.321 | 0.250 | 0.451 | 0.216 || 10.2 | 1.00 |   6.3s ||  nan |  nan
Gemma 12B                    | 0.314 | 0.239 | 0.456 | 0.205 || 10.1 | 1.00 |  14.1s || 0.05 | 0.56
GPT-4.1 nano                 | 0.310 | 0.229 | 0.480 | 0.201 ||  8.8 | 1.00 |   4.6s ||  nan |  nan
Claude Haiku 4.5 (high)      | 0.296 | 0.263 | 0.337 | 0.226 || 14.4 | 1.00 |  15.9s || 0.00 | 0.00
InternVL 3.5 2B              | 0.294 | 0.207 | 0.505 | 0.179 ||  7.5 | 1.00 | 117.4s ||  nan |  nan
Claude Haiku 4.5             | 0.289 | 0.248 | 0.345 | 0.210 || 13.2 | 1.00 |   4.3s || 0.00 | 0.00
Gemma 4B                     | 0.282 | 0.206 | 0.448 | 0.179 ||  8.5 | 1.00 |   3.2s || 0.00 | 0.00
```

</details>


<details open>
<summary>Weighted Datasets</summary>

```bash
Model                        |  F1   |  Rec. | Prec. |  mAP  || Ins. | Mat. |  Time  || Fail | Ret.
---------------------------------------------------------------------------------------------------
Gemini 2.5 Pro (high)        | 0.500 | 0.452 | 0.560 | 0.352 || 13.1 | 0.75 |  16.1s ||  nan |  nan
Gemini 2.5 Pro               | 0.500 | 0.452 | 0.560 | 0.350 || 13.0 | 0.75 |  15.3s ||  nan |  nan
Gemini 2.5 Pro (preview)     | 0.494 | 0.444 | 0.557 | 0.353 || 12.7 | 0.76 |  14.1s ||  nan |  nan
Gemini 2.5 Flash (low)       | 0.477 | 0.437 | 0.524 | 0.335 || 13.5 | 0.75 |   6.4s ||  nan |  nan
Grok 4                       | 0.476 | 0.458 | 0.500 | 0.340 || 13.4 | 0.80 |  46.4s ||  nan |  nan
Gemini 2.5 Flash Lite (high) | 0.466 | 0.395 | 0.569 | 0.303 || 10.5 | 0.79 |  10.7s || 0.00 | 0.03
Gemini 2.5 Flash (high)      | 0.465 | 0.436 | 0.499 | 0.325 || 14.1 | 0.75 |   7.5s ||  nan |  nan
InternVL 3.5 38B (think)     | 0.464 | 0.398 | 0.560 | 0.315 || 10.8 | 0.78 |  58.7s ||  nan |  nan
GLM 4.5V (think)             | 0.463 | 0.379 | 0.599 | 0.305 ||  9.3 | 0.80 |  14.9s ||  nan |  nan
GPT-5 mini (high)            | 0.461 | 0.438 | 0.487 | 0.319 || 15.2 | 0.73 |  82.6s ||  nan |  nan
Gemini 2.5 Flash (medium)    | 0.460 | 0.426 | 0.499 | 0.317 || 14.0 | 0.74 |   6.9s ||  nan |  nan
GPT-5 mini                   | 0.460 | 0.445 | 0.476 | 0.332 || 14.9 | 0.75 |  30.3s ||  nan |  nan
GPT-4.1                      | 0.458 | 0.414 | 0.513 | 0.319 || 12.1 | 0.79 |   8.1s ||  nan |  nan
Gemini 2.5 Flash             | 0.457 | 0.399 | 0.537 | 0.313 || 11.1 | 0.79 |   3.2s ||  nan |  nan
GPT-4.1 mini                 | 0.457 | 0.373 | 0.595 | 0.288 ||  9.0 | 0.81 |   5.2s ||  nan |  nan
Grok 4 Fast (high)           | 0.456 | 0.431 | 0.484 | 0.318 || 14.2 | 0.76 |   9.0s || 0.01 | 0.44
Gemini 2.5 Flash (preview)   | 0.454 | 0.393 | 0.538 | 0.310 || 11.1 | 0.79 |   3.6s ||  nan |  nan
GPT-5                        | 0.452 | 0.458 | 0.448 | 0.331 || 16.8 | 0.74 |  50.9s ||  nan |  nan
OVIS 2.5 9B (think)          | 0.452 | 0.356 | 0.620 | 0.291 ||  9.0 | 0.77 |  28.0s ||  nan |  nan
Qwen3-VL 235B A22B (think)   | 0.452 | 0.394 | 0.531 | 0.305 || 10.5 | 0.83 |   6.6s || 0.00 | 0.14
Qwen3-VL 30B A3B (think)     | 0.449 | 0.341 | 0.661 | 0.267 ||  7.1 | 0.85 |  27.1s || 0.00 | 0.12
Qwen3-VL 235B A22B           | 0.449 | 0.388 | 0.538 | 0.304 || 10.3 | 0.81 |   7.1s || 0.01 | 0.08
GPT-5 (high)                 | 0.447 | 0.462 | 0.433 | 0.330 || 17.8 | 0.73 |  95.0s ||  nan |  nan
Grok 4 Fast                  | 0.441 | 0.418 | 0.467 | 0.306 || 14.3 | 0.76 |   8.4s || 0.02 | 0.35
GLM 4.5V                     | 0.439 | 0.346 | 0.604 | 0.286 ||  8.6 | 0.80 |  15.9s || 0.03 | 0.20
GPT-5 nano (high)            | 0.435 | 0.367 | 0.533 | 0.289 || 10.8 | 0.76 |  47.7s ||  nan |  nan
InternVL 3.5 30B A3B (think) | 0.434 | 0.359 | 0.553 | 0.274 || 10.1 | 0.76 |  38.6s ||  nan |  nan
InternVL 3.5 4B (think)      | 0.432 | 0.358 | 0.550 | 0.269 ||  9.6 | 0.78 |  35.5s ||  nan |  nan
GPT-5 nano                   | 0.429 | 0.361 | 0.530 | 0.278 || 10.5 | 0.77 |  26.5s ||  nan |  nan
InternVL 3.5 30B A3B         | 0.421 | 0.332 | 0.584 | 0.264 ||  7.9 | 0.82 |  38.7s ||  nan |  nan
Mistral Medium 3.1 (t=0.15)  | 0.412 | 0.352 | 0.499 | 0.278 || 10.4 | 0.79 |   3.3s ||  nan |  nan
Qwen-VL Plus                 | 0.409 | 0.314 | 0.589 | 0.249 ||  8.0 | 0.79 |   3.6s || 0.01 | 0.44
Mistral Medium 3.1 (t=1.00)  | 0.408 | 0.346 | 0.497 | 0.280 || 11.0 | 0.78 |   3.4s ||  nan |  nan
Claude Sonnet 4.5            | 0.408 | 0.326 | 0.544 | 0.260 ||  9.1 | 0.79 |   6.3s || 0.00 | 0.00
Claude Sonnet 4.5 (high)     | 0.407 | 0.343 | 0.501 | 0.258 || 10.4 | 0.79 |  17.0s || 0.00 | 0.01
InternVL 3.5 2B (think)      | 0.403 | 0.323 | 0.540 | 0.257 ||  8.8 | 0.79 |  30.4s ||  nan |  nan
Gemma 27B                    | 0.402 | 0.343 | 0.489 | 0.256 || 11.1 | 0.75 |  10.5s || 0.00 | 0.00
Claude Sonnet 4              | 0.397 | 0.311 | 0.548 | 0.253 ||  9.1 | 0.75 |   5.9s ||  nan |  nan
Gemini 2.5 Flash Lite        | 0.396 | 0.359 | 0.453 | 0.272 || 10.9 | 0.83 |   3.0s || 0.00 | 0.01
Gemma 12B                    | 0.388 | 0.312 | 0.517 | 0.248 ||  9.7 | 0.75 |  13.3s || 0.02 | 0.47
Qwen3-VL 30B A3B             | 0.381 | 0.326 | 0.485 | 0.245 ||  9.6 | 0.83 |   7.8s || 0.05 | 0.55
InternVL 3.5 2B              | 0.380 | 0.297 | 0.547 | 0.234 ||  7.2 | 0.84 | 121.1s ||  nan |  nan
Claude Haiku 4.5 (high)      | 0.378 | 0.330 | 0.444 | 0.242 || 12.7 | 0.73 |  15.3s || 0.00 | 0.00
Claude Haiku 4.5             | 0.374 | 0.314 | 0.464 | 0.233 || 11.5 | 0.73 |   4.0s || 0.00 | 0.00
Qwen3-VL 8B                  | 0.370 | 0.339 | 0.443 | 0.264 ||  9.8 | 0.91 |   5.2s || 0.01 | 0.05
GPT-4.1 nano                 | 0.363 | 0.285 | 0.505 | 0.216 ||  7.6 | 0.85 |   4.0s ||  nan |  nan
Gemma 4B                     | 0.356 | 0.265 | 0.544 | 0.217 ||  7.4 | 0.78 |   3.2s || 0.00 | 0.01
Qwen3-VL 8B (think)          | 0.330 | 0.232 | 0.579 | 0.194 ||  8.8 | 0.80 |  84.7s || 0.31 | 1.24
```

</details>

<details>
<summary>Details</summary>

Models are sorted in descending order by F-1 score.<br>

Legend:
- F1: The achieved F-1 score of detections that passed label matching compared to groundtruth annotations.<br>
- Rec.: The achieved recall score of detections that passed label matching compared to groundtruth annotations.<br>
- Prec.: The achieved precision score of detections that passed label matching compared to groundtruth annotations.<br>
- mAP: The achieved mAP score of detections that passed label matching compared to groundtruth annotations.<br>
- Ins.: The average number of object instances in a valid structured description per image.<br>
- Mat.: The ratio of matched detections by the label matching procedure over all detections.<br>
- Time: The median time to generate a valid structured description over all images.<br>
- Fail: The rate of invalid structured descriptions after all (4) generation attempts.<br>
- Ret.: The average number of retry attempts to generate a valid structured description per image (0 to 3).<br>

Remarks:
- Models where the last two columns report nan were evaluated with an infinite and untracked number of retry attempts, until a valid structured description was obtained.<br>
- All models were used and interpreted at best effort, limiting parallel usage, attempting to extract JSON from within markdown tags or reasoning content, etc.<br>
- Reasons for failed attempts may include rate limits, content moderation, timeouts, reaching max. token limits, etc.<br>
- All reported times may heavily be affected by the used hardware, rate limits, server load, etc.

</details>

### Setup

#### Vision Models

Follow the documentation of [NimbRo Vision Servers](https://github.com/AIS-Bonn/nimbro_vision_servers) to serve the vision models you plan on using, e.g. [MM Grounding DINO](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/mmgroundingdino) or [SAM2-Realtime](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/sam2_realtime).

#### Python

To install all required [Python dependencies](./requirements.txt):
```bash
pip install -r requirements.txt
```
To access all features and improve processing speed, also install the [Python dependencies of NimbRo Utils](https://github.com/AIS-Bonn/nimbro_utils/blob/main/requirements.txt).

#### ROS2 Jazzy

Include this repository together with [NimbRo API](https://github.com/AIS-Bonn/nimbro_api), [NimbRo API Interfaces](https://github.com/AIS-Bonn/nimbro_api_interfaces) and [NimbRo Utilities](https://github.com/AIS-Bonn/nimbro_utils) in the source folder of your colcon workspace. After building them:
```bash
colcon build --packages-select nimbro_utils nimbro_api_interfaces nimbro_api vlm_gist --symlink-install
```
and re-sourcing:
```bash
source install/local_setup.bash
```
and launching NimbRo API:
```bash
ros2 launch nimbro_api launch.py
```
several [nodes](./vlm_gist/fiftyone) to interact (download,  describe, detect, validate, evaluate, label match, etc.) with [FiftyOne datasets](https://docs.voxel51.com/user_guide/using_datasets.html), and [node extensions](./vlm_gist/fiftyone) for applying the VLM-GIST pipeline, stages thereof, or baseline methods.

See the provided Jupyter Notebooks for [reproducing results](./notebooks/evaluation/command_builders.ipynb) or [using the node extensions](./notebooks/method).

### Docker

Alternatively, you can use the provided [devcontainer](./.devcontainer) or [Dockerfile](./docker/Dockerfile).

### Citation

If you utilize this package in your research, please cite:

https://arxiv.org/abs/2503.16538
```bibtex
@article{paetzold25vlmgist,
    author={Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
    title={Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
    journal={IEEE Robotics and Automation Letters (RA-L)},
    volume={10},
    number={11},
    pages={11578-11585},
    year={2025}
}
```

### License

`vlm_gist` (code, scripts, etc.) is licensed under BSD‑3.

The custom dataset (see [dataset](./data/datasets/vlm_gist)) is licensed under [CC BY‑NC‑SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (see [license](./data/datasets/vlm_gist/LICENSE)), where 42 images ([00013.jpg](./data/datasets/vlm_gist/data/00013.jpg) to [00054.jpg](./data/datasets/vlm_gist/data/00054.jpg)) are taken from the [AgiBot World](https://agibot-world.com/) dataset.

### Contact

Bastian Pätzold <paetzold@ais.uni-bonn.de><br>
Jan Nogga <nogga@ais.uni-bonn.de>
