"""For PyTabKit, we have to map names from the paper https://arxiv.org/pdf/2407.04491 (see Appendix C.3.1) to
the functions used to get data from UCI in the code (https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/bench/data/get_uci.py).
"""


""""Mapping tuples for classification"""
clf_mapping = [
    ("abalone", "get_abalone()"),
    ("adult", "get_census_income()"),
    ("anuran_calls_families", "get_anuran_calls()"),
    ("anuran_calls_genus", "get_anuran_calls()"),
    ("anuran_calls_species", "get_anuran_calls()"),
    ("avila", "get_avila()"),
    ("bank_marketing", "get_bank_marketing()"),
    ("bank_marketing_additional", "get_bank_marketing()"),
    ("chess", "get_chess()"),
    ("chess_krvk", "get_chess_krvk()"),
    ("crowd_sourced_mapping", "get_crowd_sourced_mapping()"),
    ("default_credit_card", "get_default_credit_card()"),
    ("eeg_eye_state", "get_eeg_eye_state()"),
    ("electrical_grid_stability_simulated", "get_electrical_grid_stability_simulated()"),
    ("facebook_live_sellers_thailand_status", "get_facebook_live_sellers_thailand()"),
    ("firm_teacher_clave", "get_firm_teacher_clave()"),
    ("first_order_theorem_proving", "get_first_order_theorem_proving()"),
    ("gas_sensor_drift_class", "get_gas_sensor_drift()"),
    ("gesture_phase_segmentation_raw", "get_gesture_phase_segmentation()"),
    ("gesture_phase_segmentation_va3", "get_gesture_phase_segmentation()"),
    ("htru2", "get_htru2()"),
    ("human_activity_smartphone", "get_human_activity_smartphone()"),
    ("indoor_loc_building", "get_indoor_user_movement_prediction()"),
    ("indoor_loc_relative", "get_indoor_user_movement_prediction()"),
    ("insurance_benchmark", "get_insurance_benchmark()"),
    ("landsat_satimage", "get_landsat_satimage()"),
    ("letter_recognition", "get_letter_recognition()"),
    ("madelon", "get_madelon()"),
    ("magic_gamma_telescope", "get_magic_gamma_telescope()"),
    ("mushroom", "get_mushroom()"),
    ("musk", "get_musk()"),
    ("nomao", "get_nomao()"),
    ("nursery", "get_nursery()"),
    ("occupancy_detection", "get_occupancy_detection()"),
    ("online_shoppers_attention", "get_online_shoppers_attention()"),
    ("optical_recognition_handwritten_digits", "get_optical_recognition_handwritten_digits()"),
    ("ozone_level_1hr", "get_ozone_level()"),
    ("ozone_level_8hr", "get_ozone_level()"),
    ("page_blocks", "get_page_blocks()"),
    ("pen_recognition_handwritten_characters", "get_pen_recognition_handwritten_characters()"),
    ("phishing", "get_phishing()"),
    ("polish_companies_bankruptcy_1year", "get_polish_companies_bankruptcy()"),
    ("polish_companies_bankruptcy_2year", "get_polish_companies_bankruptcy()"),
    ("polish_companies_bankruptcy_3year", "get_polish_companies_bankruptcy()"),
    ("polish_companies_bankruptcy_4year", "get_polish_companies_bankruptcy()"),
    ("polish_companies_bankruptcy_5year", "get_polish_companies_bankruptcy()"),
    ("seismic_bumps", "get_seismic_bumps()"),
    ("skill_craft", "get_skill_craft()"),
    ("smartphone_human_activity", "get_smartphone_human_activity()"),
    ("smartphone_human_activity_postural", "get_smartphone_human_activity_postural()"),
    ("spambase", "get_spambase()"),
    ("superconductivity_class", "get_superconductivity()"),
    ("thyroid_all_bp", "get_thyroids()"),
    ("thyroid_all_hyper", "get_thyroids()"),
    ("thyroid_all_hypo", "get_thyroids()"),
    ("thyroid_all_rep", "get_thyroids()"),
    ("thyroid_ann", "get_thyroids()"),
    ("thyroid_dis", "get_thyroids()"),
    ("thyroid_hypo", "get_thyroids()"),
    ("thyroid_sick", "get_thyroids()"),
    ("thyroid_sick_eu", "get_thyroids()"),
    ("turkiye_student_evaluation", "get_turkiye_student_evaluation()"),
    ("wall_follow_robot_2", "get_wall_following_robot()"),
    ("wall_follow_robot_24", "get_wall_following_robot()"),
    ("wall_follow_robot_4", "get_wall_following_robot()"),
    ("waveform", "get_waveform()"),
    ("waveform_noise", "get_waveform()"),
    ("wilt", "get_wilt()"),
    ("wine_quality_all", "get_wine_quality()"),
    ("wine_quality_type", "get_wine_quality()"),
    ("wine_quality_white", "get_wine_quality()"),
]

""""Mapping tuples for regression"""
reg_mapping = [
    ("air_quality_bc", "get_air_quality()"),
    ("air_quality_co2", "get_air_quality()"),
    ("air_quality_no2", "get_air_quality()"),
    ("air_quality_nox", "get_air_quality()"),
    ("appliances_energy", "get_appliances_energy()"),
    ("bejing_pm25", "get_bejing_pm25()"),
    ("bike_sharing_casual", "get_bike_sharing()"),
    ("bike_sharing_total", "get_bike_sharing()"),
    ("carbon_nanotubes_u", "get_carbon_nanotubes()"),
    ("carbon_nanotubes_v", "get_carbon_nanotubes()"),
    ("carbon_nanotubes_w", "get_carbon_nanotubes()"),
    ("chess_krvk", "get_chess_krvk()"),
    ("cycle_power_plant", "get_cycle_power_plant()"),
    ("electrical_grid_stability_simulated", "get_electrical_grid_stability_simulated()"),
    ("facebook_comment_volume", "get_facebook_comment_volume()"),
    ("facebook_live_sellers_thailand_shares", "get_facebook_live_sellers_thailand()"),
    ("five_cities_beijing_pm25", "get_five_cities_pm25()"),
    ("five_cities_chengdu_pm25", "get_five_cities_pm25()"),
    ("five_cities_guangzhou_pm25", "get_five_cities_pm25()"),
    ("five_cities_shanghai_pm25", "get_five_cities_pm25()"),
    ("five_cities_shenyang_pm25", "get_five_cities_pm25()"),
    ("gas_sensor_drift_class", "get_gas_sensor_drift()"),
    ("gas_sensor_drift_conc", "get_gas_sensor_drift()"),
    ("indoor_loc_alt", "get_indoor_loc()"),
    ("indoor_loc_lat", "get_indoor_loc()"),
    ("indoor_loc_long", "get_indoor_loc()"),
    ("insurance_benchmark", "get_insurance_benchmark()"),
    ("metro_interstate_traffic_volume_long", "get_metro_interstate_traffic_volume()"),
    ("metro_interstate_traffic_volume_short", "get_metro_interstate_traffic_volume()"),
    ("naval_propulsion_comp", "get_naval_propulsion()"),
    ("naval_propulsion_turb", "get_naval_propulsion()"),
    ("nursery", "get_nursery()"),
    ("online_news_popularity", "get_online_news_popularity()"),
    ("parking_birmingham", "get_parking_birmingham()"),
    ("parkinson_motor", "get_parkinson()"),
    ("parkinson_total", "get_parkinson()"),
    ("protein_tertiary_structure", "get_protein_tertiary_structure()"),
    ("skill_craft", "get_skill_craft()"),
    ("sml2010_dining", "get_sml2010()"),
    ("sml2010_room", "get_sml2010()"),
    ("superconductivity", "get_superconductivity()"),
    ("travel_review_ratings", "get_tarvel_review_ratings()"),
    ("wall_follow_robot_2", "get_wall_following_robot()"),
    ("wall_follow_robot_24", "get_wall_following_robot()"),
    ("wall_follow_robot_4", "get_wall_following_robot()"),
    ("wine_quality_all", "get_wine_quality()"),
    ("wine_quality_white", "get_wine_quality()"),
]

"""Mapping of each function to DOI from dataset from UCI"""
functions_to_dois_map = [
    ('get_abalone', '10.24432/C55C7W'),
    ('get_anuran_calls', "10.24432/C5CC9H"),
    ('get_avila', "10.24432/C5K02X"),
    ('get_bank_marketing', "10.24432/C5K306"),
    ('get_chess', "10.24432/C5DK5C"),
    ('get_chess_krvk', "10.24432/C5DK5C"),
    ("get_crowd_sourced_mapping", "10.24432/C56315"),
    ('get_default_credit_card', "10.24432/C55S3H"),
    ('get_eeg_eye_state', "10.24432/C57G7J"),
    ('get_electrical_grid_stability_simulated', "10.24432/C5PG66"),
    ('get_facebook_live_sellers_thailand', "10.24432/C5R60S"),
    ('get_firm_teacher_clave', "10.24432/C5GC9F"),
    ("get_first_order_theorem_proving", "10.24432/C5RC9X"),
    ("get_gas_sensor_drift", "10.24432/C5MK6M"),
    ("get_gesture_phase_segmentation", "10.24432/C5Z32C"),
    ("get_htru2", "10.24432/C5DK6R"),
    ("get_human_activity_smartphone", "10.24432/C54S4K"),
    ("get_indoor_user_movement_prediction", "10.24432/C5761H"),
    ("get_insurance_benchmark", "10.24432/C5630S"),
    ("get_landsat_satimage", "10.24432/C55887"), # true origin in pytabkit 10.24432/C5XS3B
    ("get_letter_recognition", "10.24432/C5ZP40"),
    ("get_madelon", "10.24432/C5602H"),
    ("get_magic_gamma_telescope", "10.24432/C52C8B"),
    ("get_mushroom", "10.24432/C5959T"),
    ("get_musk", "10.24432/C51608"),
    ("get_nomao", "10.24432/C53G79"),
    ("get_nursery", "10.24432/C5P88W"),
    ("get_occupancy_detection", "10.24432/C5X01N"),
    ("get_online_shoppers_attention", "10.24432/C5F88Q"),
    ("get_optical_recognition_handwritten_digits", "10.24432/C50P49"),
    ("get_ozone_level", "10.24432/C5NG6W"),
    ("get_page_blocks", "10.24432/C5J590"),
    ("get_pen_recognition_handwritten_characters", "10.24432/C5MG6K"),
    ("get_phishing", "10.24432/C51W2X"),
    ("get_polish_companies_bankruptcy", "10.24432/C5F600"),
    ("get_seismic_bumps", "10.24432/C5W902"),
    ("get_skill_craft", "10.24432/C5161N"),
    ("get_smartphone_human_activity", "10.24432/C5P597"),
    ("get_smartphone_human_activity_postural", "10.24432/C54G7M"),
    ("get_spambase", "10.24432/C53G6X"),
    ("get_superconductivity", "10.24432/C53P47"),
    ("get_thyroids", "10.24432/C5D010"),
    ("get_turkiye_student_evaluation", "10.24432/C5S02S"),
    ("get_wall_following_robot", "10.24432/C57C8W"),
    ("get_waveform", "10.24432/C56014"), # 10.24432/C56014 or 10.24432/C5CS3C
    ("get_wilt", "10.24432/C5KS4M"),
    ("get_wine_quality", "10.24432/C56S3T"),
    ("get_air_quality", "10.24432/C59K5F"),
    ("get_appliances_energy", "10.24432/C5VC8G"),
    ("get_census_income", "10.24432/C5XW20"),
    ("get_five_cities_pm25", "10.24432/C52K58"),
    ("get_bejing_pm25", "10.24432/C5JS49"),
    ("get_indoor_loc", "10.24432/C5D311"),
    ("get_naval_propulsion", "10.24432/C5K31K"),
    ("get_parking_birmingham", "10.24432/C51K5Z"),
    ("get_tarvel_review_ratings", "10.24432/C5C31Q"),
    ("get_sml2010", "10.24432/C5RS3S"),
    ("get_facebook_comment_volume", "10.24432/C5Q886"),
    ("get_carbon_nanotubes", "10.24432/C50892"),
    ("get_metro_interstate_traffic_volume", "10.24432/C5X60B"),
    ("get_bike_sharing", "10.24432/C5W894"),
    ("get_online_news_popularity", "10.24432/C5NS3V"),
    ("get_parkinson", "10.24432/C59C74"),
    ("get_cycle_power_plant", "10.24432/C5002N"),
    ("get_protein_tertiary_structure", "10.24432/C5QW3H"),
]

# Filter duplicates based on DOIs
dois_in_sheet = [
    # DOI, dataset_id, dataset_name
    '10.24432/C55C7W', # 42726  abalone
    "10.24432/C5K306", # 1461	bank-marketing
    "10.24432/C5DK5C", # 3	    kr-vs-kp
    "10.24432/C57G7J", # 1471	eeg-eye-state
    "10.24432/C5RC9X", # 1475	first-order-theorem-proving
    "10.24432/C5Z32C", # 4538	GesturePhaseSegmentationProcessed
    "10.24432/C54S4K", # 1478	har
    "10.24432/C55887", # 182	satimage
    "10.24432/C5602H", # 1485	madelon
    "10.24432/C52C8B", # 1120	MagicTelescope
    "10.24432/C5959T", # 24	    mushroom
    "10.24432/C51608", # 1116	musk
    "10.24432/C53G79", # 1486	nomao
    "10.24432/C5P88W", # 26	    nursery
    "10.24432/C50P49", # 28	    optdigits
    "10.24432/C5NG6W", # 1487	ozone-level-8hr
    "10.24432/C5J590", # 30	    page-blocks
    "10.24432/C5MG6K", # 32	    pendigits
    "10.24432/C51W2X", # 4534	PhishingWebsites
    "10.24432/C53G6X", # 44	    spambase
    "10.24432/C53P47", # 44964	superconductivity
    "10.24432/C5D010", # 57	    hypothyroid
    "10.24432/C57C8W", # 1497	wall-robot-navigation
    "10.24432/C56014", # 60	    waveform-5000
    "10.24432/C5KS4M", # 40983	wilt
    "10.24432/C56S3T", # 287	wine_quality
    "10.24432/C5XW20", # 1590	adult
    "10.24432/C5K31K", # 44969	naval_propulsion_plant
    "10.24432/C5NS3V", # 42724	OnlineNewsPopularity
]

# Test correctness
assert len(clf_mapping) == 71
assert len(reg_mapping) == 47

all_functions = set([x[1].replace("()","") for x in clf_mapping + reg_mapping])
functions_in_mappings = set([x[0] for x in functions_to_dois_map])
assert sorted(all_functions) == sorted(functions_in_mappings)

# Filter duplicates
functions_to_dois_map = [(a, b) for a, b in functions_to_dois_map if b not in dois_in_sheet]

# Check correct state from sheet
new_dois_from_pytabkit_in_sheet  = [
    "https://doi.org/10.24432/C5CC9H",
    "https://doi.org/10.24432/C5K02X",
    "https://doi.org/10.24432/C56315",
    "https://doi.org/10.24432/C55S3H",
    "https://doi.org/10.24432/C5PG66",
    "https://doi.org/10.24432/C5R60S",
    "https://doi.org/10.24432/C5GC9F",
    "https://doi.org/10.24432/C5MK6M",
    "https://doi.org/10.24432/C5DK6R",
    "https://doi.org/10.24432/C5761H",
    "https://doi.org/10.24432/C5630S",
    "https://doi.org/10.24432/C5ZP40",
    "https://doi.org/10.24432/C5X01N",
    "https://doi.org/10.24432/C5F88Q",
    "https://doi.org/10.24432/C5F600",
    "https://doi.org/10.24432/C5W902",
    "https://doi.org/10.24432/C5161N",
    "https://doi.org/10.24432/C5P597",
    "https://doi.org/10.24432/C54G7M",
    "https://doi.org/10.24432/C59K5F",
    "https://doi.org/10.24432/C5S02S",
    "https://doi.org/10.24432/C5VC8G",
    "https://doi.org/10.24432/C52K58",
    "https://doi.org/10.24432/C5JS49",
    "https://doi.org/10.24432/C5D311",
    "https://doi.org/10.24432/C51K5Z",
    "https://doi.org/10.24432/C5C31Q",
    "https://doi.org/10.24432/C5RS3S",
    "https://doi.org/10.24432/C5Q886",
    "https://doi.org/10.24432/C50892",
    "https://doi.org/10.24432/C5X60B",
    "https://doi.org/10.24432/C5W894",
    "https://doi.org/10.24432/C59C74",
    "https://doi.org/10.24432/C5002N",
    "https://doi.org/10.24432/C5QW3H"
]
assert sorted(new_dois_from_pytabkit_in_sheet) == sorted([f"https://doi.org/{doi}" for _, doi in functions_to_dois_map])

