srch_id:
useless for prediction

date_time:
might be relevant for prediction

site_id:
relevancy not immediately clear

visitor_location_country_id:
might be relevant for prediction

visitor_hist_starrating:
relevant (in combination)

visitor_hist_adr_usd:
relevant (look at relative price difference)

prop_country_id:
might be relevant for prediction

prop_id:
perhaps relevant for a number of properties, engineer a feature

prop_starrating:
relevant (in combination)

prop_review_score:
relevant

prop_brand_bool:
relevant

prop_location_score:
Do they capture the same element? If selected, should they be used together, combined, or just one?

comp_rate:
Lots of null values indicating that no score is present for competitor. Useful information. Possible transformations:
1) aggregate
2) count values not null
3) counting the number of negative values: how many are reporting lower prices
- engineered: combine 2 and 3, see how expedia does relative to competitors

comp_inv
- If there was no availability at the competitor, is the rate still important?

rate_percent_diff:
- tells something about how much cheaper/more expensive expedia is.

##  notes per column / checklist
srch_id: OK
date_time: split, subsequently drop date_time
site_id: OK
visitor_location_country_id: OK
visitor_hist_starrating: 94.92% missing values, what to do? new column with (absolute) diff prop_starrating
visitor_hist_adr_usd (94.9% missing values): what to do? new column with relative diff with price_usd
prop_country_id: OK
prop_id: OK
prop_starrating: OK, created bool
prop_review_score: OK, filled nan and created bool
prop_brand_bool: OK
prop_location_score1: OK
prop_location_score2: OK, missing values predicted with linear regression. Negative values set to zero
prop_log_historical_price: OK, created bool. Filled empty with mean per prop_id
price_usd: OK, removed outliers with function and cut into bins
promotion_flag: OK
srch_destination_id: OK
srch_length_of_stay: NOT DONE, remove outliers?
srch_booking_window: NOT DONE, remove outliers?
srch_adults_count: NOT DONE, combine with children count? to create something like: "family size"
srch_children_count: NOT DONE, 
srch_room_count: NOT DONE, 
srch_saturday_night_bool: OK
srch_query_affinity_score:  
orig_destination_distance: WIP
random_bool: OK
competitors:

## Random Ideas:
Ratio between count of guest and srch room count. Maybe some hotels are booked more often when people book a room for a groups. 
For example, people probably wont book a room in the Hilton if they want a family room. Hilton guests are, I think, more likely to 
be business travellers or couples.
