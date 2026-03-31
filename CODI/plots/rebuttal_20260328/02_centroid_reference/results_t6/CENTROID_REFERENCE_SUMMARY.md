# Centroid Reference Analysis

This report uses the wording `trajectory-level global reference / geometric anchor`.

## Probing Summary

| run | num_samples | accuracy | radius_mean_mean | radius_mean_std | cos_mu_z1_mean | cos_mu_z1_std | cos_mu_z1_abs_spearman_len | cos_mu_z1_slope_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simcon | 1319 | 0.5360 | 12.3391 | 1.7722 | 0.6840 | 0.0716 | 0.1972 | 0.0008 |
| simcon_sircl | 1319 | 0.5656 | 8.3809 | 1.1636 | 0.6045 | 0.0776 | 0.1136 | 0.0004 |
| codi | 1319 | 0.5345 | 10.2634 | 1.2917 | 0.8281 | 0.0597 | 0.0342 | 0.0002 |
| codi_sircl | 1319 | 0.5603 | 5.0464 | 0.8373 | 0.6916 | 0.0849 | 0.0455 | 0.0003 |

## Offline Intervention Summary

| run | anchor_mode | token_anchor_dist_mean | token_anchor_dist_std | z1_anchor_cos_mean | z1_anchor_cos_std |
| --- | --- | --- | --- | --- | --- |
| simcon | own | 12.3391 | 1.7722 | 0.6840 | 0.0716 |
| simcon | shuffled | 22.7616 | 2.8725 | 0.4024 | 0.1101 |
| simcon | wrong | 22.9205 | 3.1458 | 0.4076 | 0.1113 |
| simcon_sircl | own | 8.3809 | 1.1636 | 0.6045 | 0.0776 |
| simcon_sircl | shuffled | 16.3002 | 2.7234 | 0.3499 | 0.1154 |
| simcon_sircl | wrong | 17.0610 | 2.7009 | 0.3371 | 0.1194 |
| codi | own | 10.2634 | 1.2917 | 0.8281 | 0.0597 |
| codi | shuffled | 16.1922 | 2.8950 | 0.6114 | 0.1116 |
| codi | wrong | 16.3615 | 3.0041 | 0.6151 | 0.1172 |
| codi_sircl | own | 5.0464 | 0.8373 | 0.6916 | 0.0849 |
| codi_sircl | shuffled | 14.8630 | 3.7004 | 0.4655 | 0.1187 |
| codi_sircl | wrong | 15.2904 | 3.8866 | 0.4684 | 0.1200 |