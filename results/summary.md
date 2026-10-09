| model | field | transform | held_out | seeds | test acc | test macro-F1 | held-out F1 | epochs | min/run |
|---|---|---|---|---|---|---|---|---|---|
| cnn | both | none |  | 3 | 0.634 | 0.632 ± 0.005 |  | 10.0 | 4.4 |
| cnn | input | none |  | 3 | 0.131 | 0.110 ± 0.021 |  | 9.7 | 1.5 |
| cnn | output | no_markdown |  | 3 | 0.562 | 0.554 ± 0.019 |  | 8.7 | 3.2 |
| cnn | output | none |  | 3 | 0.611 | 0.593 ± 0.018 |  | 10.3 | 4.0 |
| cnn | output | none | code | 1 | 0.627 | 0.615 | 0.564 | 8.0 | 2.0 |
| cnn | output | none | math | 1 | 0.622 | 0.615 | 0.570 | 12.0 | 4.3 |
| cnn | output | none | writing | 1 | 0.620 | 0.609 | 0.492 | 12.0 | 4.2 |
| cnn | output | words_only |  | 3 | 0.482 | 0.472 ± 0.008 |  | 8.0 | 2.8 |
| lstm | both | none |  | 1 | 0.652 | 0.645 |  | 6.0 | 51.8 |
| lstm | input | none |  | 1 | 0.117 | 0.045 |  | 3.0 | 2.0 |
| lstm | output | no_markdown |  | 1 | 0.562 | 0.556 |  | 6.0 | 42.5 |
| lstm | output | none |  | 1 | 0.638 | 0.626 |  | 6.0 | 47.5 |
| lstm | output | none | code | 1 | 0.551 | 0.525 | 0.463 | 6.0 | 28.4 |
| lstm | output | none | math | 1 | 0.590 | 0.563 | 0.521 | 6.0 | 43.0 |
| lstm | output | none | writing | 1 | 0.629 | 0.612 | 0.509 | 6.0 | 42.4 |
| lstm | output | words_only |  | 1 | 0.534 | 0.523 |  | 6.0 | 36.8 |
