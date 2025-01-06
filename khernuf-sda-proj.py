import pandas as pd
from annoy import AnnoyIndex
import numpy as np

csv_userfilm_file = "/Users/valid/Desktop/sda-proj/spotusers.csv"
userfilm_data = pd.read_csv(csv_userfilm_file, header=None, names=["user_id", "film_id"], nrows=500000)
userfilm_data = userfilm_data.drop_duplicates()

user_to_films = userfilm_data.groupby("user_id")["film_id"].apply(list).to_dict()
unique_films = sorted(set(userfilm_data["film_id"]))
film_to_index = {film: idx for idx, film in enumerate(unique_films)}
unique_users = sorted(user_to_films.keys())
user_to_index = {user: idx for idx, user in enumerate(unique_users)}

vector_size = len(unique_films)
ann = AnnoyIndex(vector_size, metric='angular')

for user_id, films in user_to_films.items():
    user_index = user_to_index[user_id]
    user_vector = np.zeros(vector_size)
    for film in films:
        user_vector[film_to_index[film]] = 1
    ann.add_item(user_index, user_vector)

ann.build(7)
ann.save("C:/Users/valid/Desktop/sda-proj/sda-proj.ann")

def recommend_for_user(user_id, n_recommendations=5):
    if user_id not in user_to_index:
        return "User not found in dataset."
    
    user_index = user_to_index[user_id]
    similar_users = ann.get_nns_by_item(user_index, n_recommendations + 1)
    similar_users = [unique_users[idx] for idx in similar_users if idx != user_index]

    recommended_films = set()
    for similar_user in similar_users:
        recommended_films.update(user_to_films[similar_user])

    watched_films = set(user_to_films[user_id])
    recommended_films.difference_update(watched_films)

    return list(recommended_films)

u_id = 663821
recommendations = recommend_for_user(u_id)
print(f"Recommendations for user {u_id}: {recommendations}")