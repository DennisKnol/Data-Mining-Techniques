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
