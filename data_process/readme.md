**step1:**
<br>
python data_processor.py --input-dir ./raw_data/ --user-action-file user_action.csv --user-profile-file user_profile.csv --poi-info-file poi_info.csv --poi-geographic-file poi_info_groupby_geographic.csv

Add `--max-actions 100` to build a small subset from the first 100 interaction rows.

**step2:**
<br>
python post_process_features.py

Each session is expanded into 3 tokens, stored reverse-chronologically as `[F, I, S]`, i.e. chronologically
`S (scenario) -> I (intention) -> F (feedback)`. Four tasks are kept:

| task | meaning | label token | label |
|:---|:---|:---|:---|
| where | destination of the current session | S | `label_poi_*` at the S position (+ `label_neg_poi_*`) |
| how | travel mode | S | `label_travel_mode` |
| when | departure time bucket | S | `label_future_travel` |
| via | POI of the next session | I | `label_poi_*` at the I position (+ `label_neg_poi_*`) |

`where`, `how` and `when` are query-style tasks predicted from the scenario token S before the
destination and feedback are observed. The POI content features sit on I and the observed travel
mode sits on F, both strictly after S in chronological order, so a causal decoder never sees the
answer it has to predict. `via` sits on I and legitimately conditions on the current destination
while predicting the next session's POI.
