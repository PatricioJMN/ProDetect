import json

# Load the JSON data from your file
with open(r'dataset\resource\scores.json', 'r') as json_file:
    data = json.load(json_file)

# Initialize two dictionaries to store well-pronounced and mispronounced audios
well_pronounced = {}
mispronounced = {}

# Loop through each audio in the JSON data
for utt_id, audio_data in data.items():
    total_score = audio_data["total"]
    
    # Classify based on the sentence-level total score
    if total_score >= 8:
        well_pronounced[utt_id] = audio_data
    else:
        mispronounced[utt_id] = audio_data

# Save the classified data into separate JSON files
with open(r'dataset\resource\well_pronounced.json', 'w') as wp_file:
    json.dump(well_pronounced, wp_file, indent=4)

with open(r'dataset\resource\mispronounced.json', 'w') as mp_file:
    json.dump(mispronounced, mp_file, indent=4)

print("Classification completed. Check well_pronounced.json and mispronounced.json")
