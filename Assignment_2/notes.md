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

##  notes per feature / checklist
srch_id: 
- OK

date_time: 
- split, subsequently drop date_time

site_id: 
- OK

visitor_location_country_id: 
- OK

visitor_hist_starrating: 
- we use the first quartile calculated by the country which the data point located in to represent the missing data
- new column with (absolute) diff prop_starrating
- OK

visitor_hist_adr_usd:
- we use the first quartile calculated by the country which the data point located in to represent the missing data
- new column with relative diff with price_usd
- new column with absolute diff with price_usd
- OK

prop_country_id: 
- OK

prop_id: 
- OK

prop_starrating: 
- created bool
- OK FOR NOW, room for improvement

prop_review_score: 
- filled nan
- created bool
- new feature: average prop review score per prop_id
- OK

prop_brand_bool: 
- OK

prop_location_score1: 
- new feature: average score per prop_id
- OK

prop_location_score2: 
- 
- new feature: average score per prop_id

prop_location_score COMBINED
- data["prop_location_score2"] + 0.0001) / (data["prop_location_score1"] + 0.0001
- OK

prop_log_historical_price: 
- created bool: sold / not sold in previous period
- new feature: np.exp 
- new feature: data["prop_price_diff"] = data["prop_historical_price"] - data["price_usd"]
- OK

price_usd: 
- new feature: data["total_price"] = data["price_usd"] * data["srch_room_count"]
- removed outliers with function  
- cut into bins

promotion_flag: 
- OK

srch_destination_id: 
- OK

srch_length_of_stay: NOT DONE, remove outliers?

srch_booking_window: NOT DONE, remove outliers?

srch_adults_count:
combine with children count? to create something like: "person count"

srch_children_count & srch_room_count:
- new feature: data["srch_person_count"] = data["srch_adults_count"] + data["srch_children_count"] 
- new feature: data["fee_per_person"] = data["total_price"] / data["srch_person_count"]
- new feature: data["guests_per_room"] = data["srch_travelers_count"] / data["srch_room_count"]

srch_saturday_night_bool: 
- OK

srch_query_affinity_score:  
- we use the first quartile calculated by the country which the data point located in to represent the missing data
- new feature: data["score2ma"] = data["srch_query_affinity_score"] * data["prop_location_score2"]

orig_destination_distance: 
- WIP

random_bool: 
- OK

competitors:

## Normalization
- price_usd
- prop_location_score1
- prop_location_score2
_ prop_review_score


## Random Ideas:
Ratio between count of guest and srch room count. Maybe some hotels are booked more often when people book a room for a groups. 
For example, people probably wont book a room in the Hilton if they want a family room. Hilton guests are, I think, more likely to 
be business travellers or couples.
