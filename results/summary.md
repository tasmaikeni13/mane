# Lion-MANE study summary


## Pilot lr sweep (4k steps)

| optimizer | lr | final val loss @4k |
|---|---|---|
| Lion-MANE | 1e-4 | 6.6992 |
| Lion-MANE | 2e-4 | 7.4122 |
| Lion-MANE | 5e-5 | 5.1283 |
| Lion-MANE | 5e-5s025 | 17.3048 |
| AdamW | 1e-3 | 4.2885 |
| AdamW | 2e-3 | 17.2481 |
| AdamW | 3e-4 | 4.8793 |
| AdamW | 6e-4 | 4.4870 |
| Lion | 1e-4 | 7.4124 |
| Lion | 2e-4 | 7.4127 |
| Lion | 5e-5 | 4.7261 |
